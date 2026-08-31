"""
PACE-ASD — Model Architecture (Protocol Section 3)

Pipeline:
  SpatialEncoder (per-frame residual MLP with LayerNorm)
    → MicrokineticEncoder (multi-scale Conv1D with GroupNorm)
      → EventSaliencyGate / Block-ESG  [disabled when use_gate=False]
        → TemporalEventTransformer     [disabled when use_transformer=False]
          → Classifier head
            → (Platt calibration temperature at inference)

Ablation variant flags on ASDMotionModel:
  use_gate        : A2 sets False  (all 300 frames to Transformer)
  block_size=1    : A3 frame-granularity gate
  top_m=120       : A3 paired with block_size=1
  use_transformer : A4 sets False  (linear head on ESG output)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── A. Skeleton Spatial Encoder ───────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class SpatialEncoder(nn.Module):
    """
    Per-frame MLP encoder with LayerNorm for stable cross-epoch statistics.

    Input : (B, T, 33, 2)  — mid-hip centred + inter-shoulder scaled.
    Output: (B, T, spatial_dim)
    """

    def __init__(self, num_landmarks: int = 33, coord_dim: int = 2,
                 spatial_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        input_dim = num_landmarks * coord_dim * 3   # 198 (pos + vel + acc)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, spatial_dim),
            nn.LayerNorm(spatial_dim),
            nn.LeakyReLU(0.1),
        )
        self.res = ResidualBlock(spatial_dim, dropout)
        self.output_proj = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim),
            nn.LayerNorm(spatial_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 33, 2)
        B, T, L, C = x.shape

        # Mask of frames that contain actual skeleton detections
        valid_mask = x.abs().sum(dim=(-2, -1)) > 1e-4   # (B, T)

        # Velocity with boundary protection (scaled to match position range)
        vel = torch.zeros_like(x)
        vel[:, 1:] = (x[:, 1:] - x[:, :-1]) * 10.0
        valid_vel = valid_mask[:, 1:] & valid_mask[:, :-1]
        vel[:, 1:][~valid_vel] = 0.0

        # Acceleration with boundary protection (scaled to match position range)
        acc = torch.zeros_like(vel)
        acc[:, 2:] = (vel[:, 2:] - vel[:, 1:-1]) * 5.0
        valid_acc = valid_mask[:, 2:] & valid_mask[:, 1:-1] & valid_mask[:, :-2]
        acc[:, 2:][~valid_acc] = 0.0

        # Concatenate: (B, T, 198)
        features = torch.cat([
            x.reshape(B, T, -1),
            vel.reshape(B, T, -1),
            acc.reshape(B, T, -1),
        ], dim=-1)

        # Encode frame-by-frame via shared MLP
        features = features.reshape(B * T, -1)
        out = self.output_proj(self.res(self.input_proj(features)))
        out = out.reshape(B, T, -1)     # (B, T, spatial_dim)

        # Zero-out representations of empty padded frames
        out = out * valid_mask.unsqueeze(-1).float()
        return out


# ── B. Microkinetic Encoder ───────────────────────────────────────────────────

class MicrokineticEncoder(nn.Module):
    """
    Multi-scale temporal feature extractor.
    Three parallel 1-D convolutions (k=1,3,5) with GroupNorm to eliminate
    batch-norm jitter on small cohorts.

    Input : (B, T, spatial_dim)
    Output: (B, T, 3 * conv1d_channels)
    """

    def __init__(self, spatial_dim: int = 256, conv1d_channels: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        gn_groups = 8
        self.branch_k1 = nn.Sequential(
            nn.Conv1d(spatial_dim, conv1d_channels, kernel_size=1),
            nn.GroupNorm(gn_groups, conv1d_channels),
            nn.LeakyReLU(0.1),
        )
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(spatial_dim, conv1d_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, conv1d_channels),
            nn.LeakyReLU(0.1),
        )
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(spatial_dim, conv1d_channels, kernel_size=5, padding=2),
            nn.GroupNorm(gn_groups, conv1d_channels),
            nn.LeakyReLU(0.1),
        )
        self.spatial_drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, spatial_dim)
        xt  = x.transpose(1, 2)                          # (B, C, T)
        out = torch.cat([
            self.branch_k1(xt),
            self.branch_k3(xt),
            self.branch_k5(xt),
        ], dim=1)                                         # (B, 3C, T)
        out = self.spatial_drop(out.unsqueeze(2)).squeeze(2)
        return out.transpose(1, 2)                        # (B, T, 3C)


# ── C. Block-Level Event Saliency Gate (Block-ESG) ───────────────────────────

class EventSaliencyGate(nn.Module):
    """
    Allocates a fixed temporal token budget (K = top_m * block_size, default 8*15=120 frames)
    by selecting the top-M most relevant contiguous blocks of L frames each. Padded blocks
    are masked so padding frames are never prioritized.

    Mechanism note on dataset dynamics:
    Given the corpus's clip-length distribution (median 124 valid frames, n=110),
    a budget of 120 frames functions as near-complete retention / modest trimming
    for roughly half the cohort (clips <= 120 frames), while performing genuine
    sparse selection (discarding a substantial fraction of frames) for the longer clips.

    Input : (B, T, D)
    Output: selected_frames (B, M*L, D)
            selected_indices (B, M*L)   — original frame positions
    """

    def __init__(self, input_dim: int, block_size: int = 15, top_m: int = 8):
        super().__init__()
        self.block_size = block_size
        self.top_m      = top_m
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )
        self._last_block_scores: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        B, T, D  = x.shape
        L, M     = self.block_size, self.top_m

        # Pad to multiple of block_size
        if T % L != 0:
            pad_len = L - (T % L)
            x       = F.pad(x, (0, 0, 0, pad_len))
        T_pad = x.shape[1]
        N     = T_pad // L                              # number of candidate blocks

        blocks     = x.view(B, N, L, D)                # (B, N, L, D)
        block_repr = blocks.mean(dim=2)                 # (B, N, D)

        # Mask blocks that are completely empty / padded
        block_activity = blocks.abs().sum(dim=(2, 3))   # (B, N)
        valid_block_mask = block_activity > 1e-4        # (B, N)

        raw_scores = self.gate(block_repr).squeeze(-1)  # (B, N)
        self._last_block_scores = raw_scores.detach()

        # Prioritize valid movement blocks by masking out padding
        masked_scores = raw_scores.masked_fill(~valid_block_mask, -1e9)

        m = min(M, N)
        _, top_idx   = torch.topk(masked_scores, m, dim=1)    # (B, m)
        top_idx, _   = torch.sort(top_idx, dim=1)            # temporal order

        # Gather selected blocks
        top_exp = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D)
        sel_blocks = torch.gather(blocks, 1, top_exp)        # (B, m, L, D)

        # Differentiable saliency weighting
        sel_scores  = torch.gather(raw_scores, 1, top_idx)                    # (B, m)
        sal_weights = torch.sigmoid(sel_scores).unsqueeze(-1).unsqueeze(-1)
        sel_blocks  = sel_blocks * sal_weights

        sel_frames = sel_blocks.reshape(B, m * L, D)

        # Reconstruct original frame indices
        offsets         = top_idx * L                                        # (B, m)
        frame_offsets   = torch.arange(L, device=x.device).view(1, 1, L)    # (1,1,L)
        sel_indices     = (offsets.unsqueeze(-1) + frame_offsets).reshape(B, -1)

        return sel_frames, sel_indices

    def get_block_scores(self) -> torch.Tensor | None:
        return self._last_block_scores


# ── D. Temporal Event Transformer ────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Non-trainable sinusoidal positional encodings (time-position aware)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, K)
        B, K = indices.shape
        pe   = self.pe.expand(B, -1, -1)              # (B, max_len, d_model)
        idx  = indices.unsqueeze(-1).expand(-1, -1, pe.shape[-1])
        return torch.gather(pe, 1, idx)               # (B, K, d_model)


class TemporalEventTransformer(nn.Module):
    """
    Processes sparse event tokens with bidirectional self-attention.
    Uses sinusoidal positional encoding (absolute frame position).
    Mean-pooled output → linear projection.

    Input : (B, K, input_dim), indices (B, K)
    Output: (B, output_dim)
    """

    def __init__(self, input_dim: int, output_dim: int = 256,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj  = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.Dropout(dropout)
        )
        self.pos_enc     = SinusoidalPositionalEncoding(input_dim)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads,
            dim_feedforward=input_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool_drop   = nn.Dropout(dropout)
        self.out_proj    = nn.Linear(input_dim, output_dim)
        self.norm        = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) + self.pos_enc(indices)
        x = self.transformer(x)
        x = self.pool_drop(x.mean(dim=1))
        return self.norm(self.out_proj(x))

    def get_attention_maps(self, x: torch.Tensor,
                           indices: torch.Tensor) -> torch.Tensor:
        """Return attention weights from the final transformer layer."""
        x = self.input_proj(x) + self.pos_enc(indices)
        for layer in self.transformer.layers[:-1]:
            x = layer(x)
        last = self.transformer.layers[-1]
        x_n  = last.norm1(x)
        _, attn = last.self_attn(x_n, x_n, x_n,
                                 need_weights=True, average_attn_weights=True)
        return attn


# ── E. Full PACE-ASD Model ────────────────────────────────────────────────────

class ASDMotionModel(nn.Module):
    """
    Full PACE-ASD pipeline with ablation variant flags.

    Args:
        config       : dict from config.yaml
        use_gate     : if False, skip ESG and feed all frames to Transformer (A2)
        use_transformer: if False, replace Transformer with linear mean-pool head (A4)
    """

    def __init__(self, config: dict,
                 use_gate: bool = True,
                 use_transformer: bool = True):
        super().__init__()
        mc           = config["model"]
        spatial_dim  = mc["spatial_dim"]
        conv1d_ch    = mc["conv1d_channels"]
        micro_dim    = 3 * conv1d_ch            # 384 with defaults
        n_heads      = mc["transformer_heads"]
        n_layers     = mc["transformer_layers"]
        dropout      = mc["dropout"]
        self.use_gate        = use_gate
        self.use_transformer = use_transformer

        self.spatial_encoder = SpatialEncoder(
            spatial_dim=spatial_dim, dropout=dropout,
        )
        self.microkinetic_encoder = MicrokineticEncoder(
            spatial_dim=spatial_dim,
            conv1d_channels=conv1d_ch,
            dropout=dropout,
        )

        if use_gate:
            self.saliency_gate = EventSaliencyGate(
                input_dim=micro_dim,
                block_size=mc.get("event_block_size", 15),
                top_m=mc.get("event_top_m", 8),
            )
        else:
            self.saliency_gate = None

        if use_transformer:
            self.temporal_transformer = TemporalEventTransformer(
                input_dim=micro_dim,
                output_dim=spatial_dim,
                n_heads=n_heads, n_layers=n_layers,
                dropout=dropout,
            )
        else:
            self.temporal_transformer = None
            self.no_tf_proj = nn.Sequential(
                nn.Linear(micro_dim, spatial_dim),
                nn.LayerNorm(spatial_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim // 2),
            nn.LayerNorm(spatial_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim // 2, 1),
        )

        # Platt calibration temperature (post-hoc)
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor, calibrate: bool = False):
        """
        Args:
            x         : (B, T, 33, 2) preprocessed skeleton sequences
            calibrate : if True, apply temperature scaling to logits

        Returns:
            probs  : (B,) sigmoid probabilities
            logits : (B,) raw logits
        """
        # A. Spatial encoding
        spatial = self.spatial_encoder(x)            # (B, T, spatial_dim)

        # B. Multi-scale temporal encoding
        micro   = self.microkinetic_encoder(spatial)  # (B, T, micro_dim)

        # C. Event selection
        if self.use_gate:
            tokens, indices = self.saliency_gate(micro)  # (B, M*L, micro_dim)
        else:
            tokens  = micro
            B, T, _ = micro.shape
            indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # D. Temporal aggregation
        if self.use_transformer:
            global_repr = self.temporal_transformer(tokens, indices)  # (B, spatial_dim)
        else:
            global_repr = self.no_tf_proj(tokens.mean(dim=1))

        # E. Classification
        logits = self.classifier(global_repr).squeeze(-1)   # (B,)

        if calibrate:
            logits = logits / self.temperature.clamp(min=1e-4)

        return torch.sigmoid(logits), logits

    def get_attention_maps(self, x: torch.Tensor):
        """Extract attention weights and saliency scores."""
        with torch.no_grad():
            spatial = self.spatial_encoder(x)
            micro   = self.microkinetic_encoder(spatial)

            if self.use_gate:
                tokens, indices = self.saliency_gate(micro)
                block_scores    = self.saliency_gate.get_block_scores()
            else:
                B, T, _ = micro.shape
                tokens  = micro
                indices = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
                block_scores = None

            if self.use_transformer:
                attn = self.temporal_transformer.get_attention_maps(tokens, indices)
            else:
                attn = None

        return attn, indices, block_scores
