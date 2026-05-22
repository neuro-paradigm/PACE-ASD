"""
ASDMotion - Full Model Architecture

Pipeline:
  Skeleton Spatial Encoder (MLP) -> Microkinetic Encoder (multi-scale Conv1D)
  -> Saliency Gate (top-K) -> Temporal Event Transformer -> Calibration -> Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ────────────────────────────────────────────────────────────
# A. Skeleton Spatial Encoder (per-frame MLP)
# ────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.act = nn.LeakyReLU(0.1)


    def forward(self, x):
        return self.act(x + self.block(x))


class SpatialEncoder(nn.Module):
    """
    Per-frame spatial encoder for skeleton data.
    Input:  (B, T, 33, 3) skeleton sequences
    Output: (B, T, spatial_dim) per-frame feature vectors

    Now uses a Residual MLP architecture for better gradient flow.
    Includes Velocity (dx/dt) and Acceleration (dv/dt) features.
    """

    def __init__(self, num_landmarks=33, coord_dim=2, spatial_dim=256, dropout=0.3):
        super().__init__()
        input_dim = num_landmarks * coord_dim * 3 # 297

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1)
        )
        
        self.res_layers = nn.Sequential(
            ResidualBlock(512, dropout)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(512, spatial_dim),
            nn.BatchNorm1d(spatial_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        # x: (B, T, 33, 3) or (B, T, 33, 2)
        # Force 2D (XY) to ensure consistency between datasets
        x = x[:, :, :, :2]
        B, T, L, C = x.shape

        # 1. Centering (Translation Invariance)
        nose = x[:, :, 0:1, :]
        x_centered = x - nose

        # 1.5 Dual-Axis Scale Normalization (Aspect-Ratio Invariance)
        left_shoulder = x_centered[:, :, 11:12, :]
        right_shoulder = x_centered[:, :, 12:13, :]
        shoulder_dist = torch.norm(left_shoulder - right_shoulder, dim=-1, keepdim=True)  # (B, T, 1, 1)
        shoulder_dist = torch.clamp(shoulder_dist, min=1e-5)
        
        # Scale X coordinates by shoulder distance; keep Y coordinates as-is
        x_norm = x_centered.clone()
        x_norm[:, :, :, 0] = x_centered[:, :, :, 0] / shoulder_dist[:, :, :, 0]

        # 2. Velocity (dx/dt)
        vel = torch.zeros_like(x_norm)
        vel[:, 1:, :, :] = x_norm[:, 1:, :, :] - x_norm[:, :-1, :, :]
        vel[:, 0, :, :] = vel[:, 1, :, :]  # Replicate boundary to avoid zero-signature cheat at t=0

        # 3. Acceleration (dv/dt)
        accel = torch.zeros_like(vel)
        accel[:, 1:, :, :] = vel[:, 1:, :, :] - vel[:, :-1, :, :]
        accel[:, 0, :, :] = accel[:, 1, :, :]  # Replicate boundary to avoid zero-signature cheat at t=0

        # 4. Concatenate Pos + Vel + Accel
        # Reshape to (B, T, 297)
        x_flat = x_norm.reshape(B, T, -1)
        v_flat = vel.reshape(B, T, -1)
        a_flat = accel.reshape(B, T, -1)
        features = torch.cat([x_flat, v_flat, a_flat], dim=-1) # (B, T, 297)

        # 5. Encode
        features = features.reshape(B * T, -1)
        x = self.input_proj(features)
        x = self.res_layers(x)
        out = self.output_proj(x)
        return out.reshape(B, T, -1)


# ────────────────────────────────────────────────────────────
# B. Microkinetic Encoder
# ────────────────────────────────────────────────────────────

class MicrokineticEncoder(nn.Module):
    """
    Multi-scale temporal feature extractor.
    Three parallel 1D convolutions at different kernel sizes capture
    instantaneous, short-range, and medium-range motion patterns.

    Input:  (B, T, spatial_dim)
    Output: (B, T, 3 * conv1d_channels)
    """

    def __init__(self, spatial_dim=256, conv1d_channels=128, dropout=0.3):
        super().__init__()

        self.branch_k1 = nn.Sequential(
            nn.Conv1d(spatial_dim, conv1d_channels, kernel_size=1),
            nn.BatchNorm1d(conv1d_channels),
            nn.LeakyReLU(0.1),
        )
        self.branch_k3 = nn.Sequential(
            nn.Conv1d(spatial_dim, conv1d_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv1d_channels),
            nn.LeakyReLU(0.1),
        )
        self.branch_k5 = nn.Sequential(
            nn.Conv1d(spatial_dim, conv1d_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv1d_channels),
            nn.LeakyReLU(0.1),
        )

        # Spatial dropout: drops entire channels (temporal consistency)
        self.spatial_dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        # x: (B, T, spatial_dim)
        x_t = x.transpose(1, 2)  # (B, spatial_dim, T) for Conv1D

        out1 = self.branch_k1(x_t)  # (B, conv1d_channels, T)
        out3 = self.branch_k3(x_t)
        out5 = self.branch_k5(x_t)

        # Concatenate: (B, 3 * conv1d_channels, T)
        out = torch.cat([out1, out3, out5], dim=1)

        # Spatial dropout (treat as 2D with fake height=1)
        out = out.unsqueeze(2)       # (B, C, 1, T)
        out = self.spatial_dropout(out)
        out = out.squeeze(2)         # (B, C, T)

        return out.transpose(1, 2)   # (B, T, 3*conv1d_channels)


# ────────────────────────────────────────────────────────────
# C. Saliency Gate
# ────────────────────────────────────────────────────────────

class SaliencyGate(nn.Module):
    """
    Learns a per-frame importance score and selects top-K frames.
    This creates a sparse set of "event tokens" for the transformer.

    Input:  (B, T, input_dim)
    Output: (B, top_k, input_dim) - selected event tokens (sorted temporally)
    """

    def __init__(self, input_dim, top_k=32):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        scores = self.gate(x).squeeze(-1)  # (B, T)
        k = min(self.top_k, x.shape[1])
        _, topk_indices = torch.topk(scores, k, dim=1)  # (B, k)
        topk_indices, _ = torch.sort(topk_indices, dim=1)  # Temporal order
        
        # Gather selected frames
        topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, x.shape[2])
        selected = torch.gather(x, 1, topk_indices_exp)  # (B, k, input_dim)
        return selected, topk_indices


# ────────────────────────────────────────────────────────────
# D. Temporal Event Transformer
# ────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Implements non-trainable, robust sinusoidal positional encodings for time-awareness."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, indices):
        # indices: (B, K) original indices from saliency gate
        B, K = indices.shape
        pe = self.pe.expand(B, -1, -1) # (B, max_len, d_model)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, pe.shape[-1]) # (B, K, d_model)
        gathered_pe = torch.gather(pe, 1, indices_expanded) # (B, K, d_model)
        return gathered_pe


class TemporalEventTransformer(nn.Module):
    """
    Processes sparse event tokens with self-attention.
    Uses bidirectional self-attention and mean pooling for stable global temporal reasoning.

    Input:  (B, K, input_dim)
    Output: (B, output_dim)
    """

    def __init__(self, input_dim, output_dim=256, n_heads=4, n_layers=2,
                 dropout=0.3):
        super().__init__()

        # Project to transformer dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(dropout),
        )

        # Sinusoidal positional encoding to prevent learned position overfitting
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=input_dim, max_len=512)

        # Transformer (Use a standard, stable dropout of 0.1 internally to prevent attention collapse)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=input_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.pool_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, indices):
        # x: (B, K, input_dim)
        # indices: (B, K) original frame indices
        B, K, _ = x.shape

        # Add sinusoidal positional encoding based on original frame positions
        target_pos_emb = self.pos_encoding(indices)
        x = self.input_proj(x) + target_pos_emb

        # Transformer encoding (no causal mask -> bidirectional temporal reasoning)
        x = self.transformer(x)  # (B, K, input_dim)

        # Global Mean Pooling to aggregate diagnostic information evenly across the entire clip
        x = x.mean(dim=1)  # (B, input_dim)
        x = self.pool_dropout(x)
        x = self.output_proj(x)  # (B, output_dim)
        x = self.norm(x)
        return x

    def get_attention_maps(self, x, indices):
        """Extract attention weights from the final transformer layer for explainability (bidirectional)."""
        B, K, _ = x.shape
        target_pos_emb = self.pos_encoding(indices)
        x = self.input_proj(x) + target_pos_emb
        
        # Pass through all but last layer without causal mask
        for layer in self.transformer.layers[:-1]:
            x = layer(x)
            
        # Manually compute attention for the last layer to get weights (bidirectional)
        last_layer = self.transformer.layers[-1]
        x_norm = last_layer.norm1(x)
        _, attn_weights = last_layer.self_attn(
            x_norm, x_norm, x_norm,
            need_weights=True,
            average_attn_weights=True
        )
        return attn_weights


# ────────────────────────────────────────────────────────────
# E. Full ASDMotion Model
# ────────────────────────────────────────────────────────────

class ASDMotionModel(nn.Module):
    """
    Full ASDMotion pipeline:
      Skeleton -> Spatial Encoder -> Microkinetic Encoder ->
      Saliency Gate -> Temporal Event Transformer -> Calibrated Prediction
    """

    def __init__(self, config):
        super().__init__()
        mc = config['model']
        spatial_dim = mc['spatial_dim']
        conv1d_ch = mc['conv1d_channels']
        micro_out_dim = 3 * conv1d_ch  # Concatenated output
        top_k = mc['top_k']
        tf_heads = mc['transformer_heads']
        tf_layers = mc['transformer_layers']
        dropout = mc['dropout']

        self.spatial_encoder = SpatialEncoder(
            spatial_dim=spatial_dim, dropout=dropout
        )
        self.microkinetic_encoder = MicrokineticEncoder(
            spatial_dim=spatial_dim, conv1d_channels=conv1d_ch,
            dropout=dropout
        )
        self.saliency_gate = SaliencyGate(
            input_dim=micro_out_dim, top_k=top_k
        )
        self.temporal_transformer = TemporalEventTransformer(
            input_dim=micro_out_dim, output_dim=spatial_dim,
            n_heads=tf_heads, n_layers=tf_layers, dropout=dropout
        )

        # Classification head with hidden layer + LayerNorm + GELU
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim // 2),
            nn.LayerNorm(spatial_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(spatial_dim // 2, 1)
        )

        # Temperature parameter for calibration (learned post-hoc)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, calibrate=False):
        """
        Args:
            x: (B, T, 33, 3) skeleton sequences
            calibrate: if True, apply temperature scaling to logits

        Returns:
            probs: (B,) sigmoid probabilities
            logits: (B,) raw logits (for BCEWithLogitsLoss)
        """
        # A. Spatial encoding (per-frame)
        spatial_features = self.spatial_encoder(x)  # (B, T, spatial_dim)

        # B. Microkinetic encoding (multi-scale temporal)
        micro_features = self.microkinetic_encoder(spatial_features)  # (B, T, 3*conv1d)

        # C. Saliency gate (sparse top-K selection with indices)
        event_tokens, event_indices = self.saliency_gate(micro_features)  # (B, K, 3*conv1d), (B, K)

        # D. Temporal transformer (aware of absolute time)
        global_repr = self.temporal_transformer(event_tokens, event_indices)  # (B, spatial_dim)

        # E. Classification
        logits = self.classifier(global_repr).squeeze(-1)  # (B,)

        if calibrate:
            logits = logits / self.temperature

        probs = torch.sigmoid(logits)
        return probs, logits

    def get_attention_maps(self, sequences):
        """
        Runs a forward pass to extract interpretable attention maps.
        Returns:
            attn_weights: (B, K, K) attention matrix showing frame-to-frame influence.
            indices: (B, K) the original frame indices that were selected by the saliency gate.
        """
        with torch.no_grad():
            spatial_feats = self.spatial_encoder(sequences)
            micro_feats = self.microkinetic_encoder(spatial_feats)
            salient_feats, indices = self.saliency_gate(micro_feats)
            attn_weights = self.temporal_transformer.get_attention_maps(salient_feats, indices)
            return attn_weights, indices
