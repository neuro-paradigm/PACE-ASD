"""
PACE-ASD — Literature Baselines (A5, Protocol Sections 4 + 8)

PyTorch models (all operate on (B, T, 33, 2) MediaPipe pose sequences):
  StackedLSTM           — 2-layer stacked LSTM
  Conv1DBiLSTMAttn      — Conv1D → BiLSTM → additive attention
  KinematicCNNLSTM      — Multi-scale CNN on pos+vel+acc → LSTM
  STTS                  — Spatial-Temporal Transformer for Skeleton
  MSG3D                 — Multi-Scale Graph 3D Conv (MediaPipe skeleton graph)
  MSG3DConvNeXt         — MSG3D + ConvNeXt temporal refinement
  SkelFormer            — Adaptive hierarchical temporal-split skeleton transformer
                          (Tier A: Section 8 — highest novelty risk)
  MTCFormer             — Multi-Grained Temporal Clip Transformer
                          (Tier A: Section 8)
  MTT                   — Multi-Scale Temporal Transformer
                          (Tier A: Section 8)
  STAR                  — Sparse Transformer-based Action Recognition
                          (Tier A: Section 8)

Scikit-learn baselines (mean-pooled flattened features):
  LR, SVM (RBF), RF, XGBoost

All baselines use identical Dryad-only splits, 20 seeds × 3-fold CV.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  MediaPipe Pose skeleton graph (33 landmarks)
# ═══════════════════════════════════════════════════════════════════════════════

_MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]
N_JOINTS = 33


def _build_adjacency(n: int = N_JOINTS, edges=_MEDIAPIPE_EDGES) -> torch.Tensor:
    """Symmetric normalised adjacency  D^{-1/2}(A+I)D^{-1/2}."""
    A = torch.zeros(n, n)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += torch.eye(n)
    D = A.sum(1).pow(-0.5)
    D[torch.isinf(D)] = 0.0
    D = torch.diag(D)
    return D @ A @ D


def _row_norm(A: torch.Tensor) -> torch.Tensor:
    row = A.sum(1, keepdim=True).clamp(min=1e-6)
    return A / row


_ADJ  = _build_adjacency()
_ADJ2 = _row_norm(_ADJ @ _ADJ)
_ADJ3 = _row_norm(_ADJ2 @ _ADJ)


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared kinematic & padding mask helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _valid_mask(x: torch.Tensor) -> torch.Tensor:
    """(B,T,33,2) → (B,T) bool — True where the frame has real (non-padded) data."""
    return x.abs().sum(dim=(-2, -1)) > 1e-4


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    x: tensor containing temporal dimension `dim` (e.g. (B, T, D) with dim=1 or (B, C, T) with dim=-1)
    mask: (B, T) bool tensor
    Computes average along `dim` only taking valid frames into account.
    """
    m = mask.float()
    while m.dim() < x.dim():
        if dim == 1:
            m = m.unsqueeze(-1)
        elif dim == -1 or dim == x.dim() - 1:
            m = m.unsqueeze(1)
        else:
            m = m.unsqueeze(-1)
    denom = m.sum(dim=dim, keepdim=True).clamp(min=1.0)
    return (x * m).sum(dim=dim) / denom.squeeze(dim)


def _kinematic(x: torch.Tensor) -> torch.Tensor:
    """
    (B,T,33,2) → (B,T,198) — position + velocity + acceleration.
    Masks derivatives across the valid/padding boundary so padding
    doesn't create a spurious motion spike at the end of each clip.
    """
    B, T, L, C = x.shape
    valid_mask = x.abs().sum(dim=(-2, -1)) > 1e-4

    vel = torch.zeros_like(x)
    vel[:, 1:] = x[:, 1:] - x[:, :-1]
    valid_vel = valid_mask[:, 1:] & valid_mask[:, :-1]
    vel[:, 1:][~valid_vel] = 0.0
    vel[:, 0] = vel[:, 1]

    acc = torch.zeros_like(vel)
    acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
    valid_acc = valid_mask[:, 1:] & valid_mask[:, :-1]
    acc[:, 1:][~valid_acc] = 0.0
    acc[:, 0] = acc[:, 1]

    return torch.cat([x, vel, acc], dim=-1).reshape(B, T, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Stacked LSTM
# ═══════════════════════════════════════════════════════════════════════════════

class StackedLSTM(nn.Module):
    """2-layer stacked LSTM with pack_padded_sequence. Input: (B, T, 33, 2)."""

    def __init__(self, input_dim=66, hidden=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, calibrate=False):
        B, T, L, C = x.shape
        lengths = _valid_mask(x).sum(dim=1).clamp(min=1).cpu()
        x_flat = x.reshape(B, T, L * C)
        packed = nn.utils.rnn.pack_padded_sequence(x_flat, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        logits = self.head(h[-1]).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Conv1D-BiLSTM-Attention
# ═══════════════════════════════════════════════════════════════════════════════

class Conv1DBiLSTMAttn(nn.Module):
    """Conv1D temporal features → BiLSTM → masked additive attention. Input: (B,T,33,2)."""

    def __init__(self, input_dim=66, conv_ch=128, hidden=128, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_ch, 3, padding=1),
            nn.BatchNorm1d(conv_ch), nn.ReLU(), nn.Dropout(dropout),
        )
        self.bilstm = nn.LSTM(conv_ch, hidden, 1, batch_first=True, bidirectional=True)
        d = hidden * 2
        self.attn = nn.Linear(d, 1)
        self.head  = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x, calibrate=False):
        B, T, L, C = x.shape
        mask = _valid_mask(x)                                   # (B, T)
        x_flat = x.reshape(B, T, L * C).transpose(1, 2)
        x_conv = self.conv(x_flat).transpose(1, 2)
        out, _ = self.bilstm(x_conv)
        scores = self.attn(out).squeeze(-1)                    # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)          # (B, T, 1)
        w = torch.nan_to_num(w, nan=0.0)
        ctx = (out * w).sum(1)                                 # (B, d)
        logits = self.head(ctx).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Kinematic CNN-LSTM
# ═══════════════════════════════════════════════════════════════════════════════

class KinematicCNNLSTM(nn.Module):
    """
    Multi-scale 1D CNN on pos+vel+acc (198 channels) → 2-layer LSTM with pack_padded_sequence.
    Three parallel branches (kernels 3, 5, 7) capture different temporal scales.
    Input: (B, T, 33, 2)
    """

    def __init__(self, cnn_ch=128, lstm_hidden=256, n_lstm=2, dropout=0.3):
        super().__init__()
        in_ch = 198
        def _branch(k):
            return nn.Sequential(
                nn.Conv1d(in_ch, cnn_ch, k, padding=k // 2), nn.BatchNorm1d(cnn_ch), nn.ReLU(),
                nn.Conv1d(cnn_ch, cnn_ch, k, padding=k // 2), nn.BatchNorm1d(cnn_ch), nn.ReLU(),
            )
        self.b3, self.b5, self.b7 = _branch(3), _branch(5), _branch(7)
        self.lstm = nn.LSTM(cnn_ch * 3, lstm_hidden, n_lstm, batch_first=True,
                            dropout=dropout if n_lstm > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(), nn.Linear(lstm_hidden // 2, 1),
        )

    def forward(self, x, calibrate=False):
        lengths = _valid_mask(x).sum(dim=1).clamp(min=1).cpu()
        f = _kinematic(x).transpose(1, 2)   # (B, 198, T)
        fused = torch.cat([self.b3(f), self.b5(f), self.b7(f)], dim=1).transpose(1, 2) # (B, T, cnn_ch*3)
        packed = nn.utils.rnn.pack_padded_sequence(fused, lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        logits = self.head(h[-1]).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  4. STTS — Spatial-Temporal Transformer for Skeleton
#     Plizzari et al., 2021 — adapted to 2D MediaPipe keypoints with padding mask.
# ═══════════════════════════════════════════════════════════════════════════════

class STTS(nn.Module):
    """
    Dual-branch skeleton transformer:
      Spatial branch  — joint-level self-attention per frame  (joints as tokens)
      Temporal branch — frame-level self-attention per joint  (frames as tokens)
    Fusion: concatenate global descriptors → head.
    Input: (B, T, 33, 2)
    """

    def __init__(self, n_joints=33, coord=2, d_model=64, n_heads=4,
                 n_sp=2, n_tp=2, dropout=0.3, max_frames=300):
        super().__init__()
        self.d = d_model
        self.n_joints = n_joints
        self.joint_emb = nn.Linear(coord, d_model)
        self.joint_pos = nn.Parameter(torch.randn(1, n_joints, d_model) * 0.02)
        self.frame_emb = nn.Embedding(max_frames, d_model)
        def _tf(n_layers):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                               dropout=0.1, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n_layers)
        self.sp_tf = _tf(n_sp)
        self.tp_tf = _tf(n_tp)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
            nn.GELU(), nn.Linear(d_model, 1),
        )

    def forward(self, x, calibrate=False):
        B, T, J, C = x.shape
        mask = _valid_mask(x)  # (B, T)
        # Spatial: joint-level self-attention per frame
        xs = self.joint_emb(x.reshape(B * T, J, C)) + self.joint_pos
        xs = self.sp_tf(xs).reshape(B, T, J, self.d).mean(dim=2)  # (B, T, d) mean over joints
        xs = _masked_mean(xs, mask, dim=1)                         # (B, d) masked mean over T

        # Temporal: frame-level self-attention per joint
        xt = self.joint_emb(x.permute(0, 2, 1, 3).reshape(B * J, T, C))
        xt = xt + self.frame_emb(torch.arange(T, device=x.device)).unsqueeze(0)
        tp_mask = (~mask).repeat_interleave(J, dim=0)             # (B*J, T)
        xt = self.tp_tf(xt, src_key_padding_mask=tp_mask).reshape(B, J, T, self.d).mean(dim=1) # (B, T, d) mean over joints
        xt = _masked_mean(xt, mask, dim=1)                         # (B, d) masked mean over T

        logits = self.head(torch.cat([xs, xt], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  5. MS-G3D — Multi-Scale Graph 3D Convolution
#     Liu et al., CVPR 2020 — simplified, MediaPipe skeleton graph.
# ═══════════════════════════════════════════════════════════════════════════════

class _MSG3DBlock(nn.Module):
    """
    One MS-G3D block: K-scale graph convolutions in parallel → depthwise temporal conv.
    branch_ch = out_ch // n_scales  →  effective_out_ch = branch_ch * n_scales
    (guaranteed divisible by groups, regardless of input out_ch).
    """

    def __init__(self, in_ch, out_ch, n_scales=3, t_kernel=9, dropout=0.2):
        super().__init__()
        self.branch_ch = max(1, out_ch // n_scales)
        self.out_ch    = self.branch_ch * n_scales
        self.g = nn.ModuleList([nn.Linear(in_ch, self.branch_ch) for _ in range(n_scales)])
        self.t = nn.Sequential(
            nn.Conv1d(self.out_ch, self.out_ch, t_kernel, padding=t_kernel // 2,
                      groups=self.out_ch),
            nn.BatchNorm1d(self.out_ch), nn.ReLU(),
        )
        self.bn   = nn.BatchNorm1d(self.out_ch)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Linear(in_ch, self.out_ch) if in_ch != self.out_ch else nn.Identity()

    def forward(self, x, adjs):
        B, T, N, C = x.shape
        res = self.res(x)
        outs = []
        for gc, adj in zip(self.g, adjs):
            xr = x.reshape(B * T, N, C)
            outs.append(F.relu(gc(adj.to(x.device) @ xr)).reshape(B, T, N, -1))
        xc = torch.cat(outs, -1)                                   # (B,T,N,out_ch)
        xt = xc.permute(0, 2, 3, 1).reshape(B * N, self.out_ch, T)
        xt = self.t(xt).reshape(B, N, self.out_ch, T).permute(0, 3, 1, 2)
        out = F.relu(self.bn((xt + res).reshape(B * T * N, self.out_ch))
                     .reshape(B, T, N, self.out_ch))
        return self.drop(out)


class MSG3D(nn.Module):
    """
    MS-G3D: 3 stacked blocks on MediaPipe skeleton graph, global masked avg-pool → head.
    Input: (B, T, 33, 2)
    """

    def __init__(self, in_ch=2, base_ch=64, dropout=0.3):
        super().__init__()
        self.register_buffer("adj1", _ADJ)
        self.register_buffer("adj2", _ADJ2)
        self.register_buffer("adj3", _ADJ3)
        self.emb = nn.Linear(in_ch, base_ch)
        b1 = _MSG3DBlock(base_ch,     base_ch * 2, dropout=dropout)
        b2 = _MSG3DBlock(b1.out_ch,   base_ch * 4, dropout=dropout)
        b3 = _MSG3DBlock(b2.out_ch,   base_ch * 4, dropout=dropout)
        self.blocks = nn.ModuleList([b1, b2, b3])
        final_ch = b3.out_ch
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(final_ch, final_ch // 2),
            nn.ReLU(), nn.Linear(final_ch // 2, 1),
        )

    def forward(self, x, calibrate=False):
        mask = _valid_mask(x)
        x = self.emb(x)
        adjs = [self.adj1, self.adj2, self.adj3]
        for blk in self.blocks:
            x = blk(x, adjs)
        x_joints = x.mean(dim=2)                             # (B, T, out_ch) mean over unpadded joints
        x_pool   = _masked_mean(x_joints, mask, dim=1)        # (B, out_ch) masked mean over T
        logits   = self.head(x_pool).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  6. MS-G3D + ConvNeXt
# ═══════════════════════════════════════════════════════════════════════════════

class _ConvNeXtBlock1D(nn.Module):
    """ConvNeXt-style block for 1-D sequences.  Input/output: (B, C, T)."""

    def __init__(self, ch, expand=4):
        super().__init__()
        self.dw  = nn.Conv1d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1  = nn.Linear(ch, ch * expand)
        self.pw2  = nn.Linear(ch * expand, ch)

    def forward(self, x):
        r = x
        x = self.norm(self.dw(x).transpose(1, 2))
        x = self.pw2(F.gelu(self.pw1(x))).transpose(1, 2)
        return x + r


class MSG3DConvNeXt(nn.Module):
    """
    MSG3D backbone → joint mean-pool to (B, ch, T) → 3 ConvNeXt blocks → masked head.
    Input: (B, T, 33, 2)
    """

    def __init__(self, in_ch=2, base_ch=64, dropout=0.3):
        super().__init__()
        self.register_buffer("adj1", _ADJ)
        self.register_buffer("adj2", _ADJ2)
        self.register_buffer("adj3", _ADJ3)
        self.emb = nn.Linear(in_ch, base_ch)
        b1 = _MSG3DBlock(base_ch,   base_ch * 2, dropout=dropout)
        b2 = _MSG3DBlock(b1.out_ch, base_ch * 4, dropout=dropout)
        self.g3d = nn.ModuleList([b1, b2])
        mid_ch = b2.out_ch
        self.cnx = nn.Sequential(*[_ConvNeXtBlock1D(mid_ch) for _ in range(3)])
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(mid_ch, mid_ch // 2),
            nn.GELU(), nn.Linear(mid_ch // 2, 1),
        )

    def forward(self, x, calibrate=False):
        mask = _valid_mask(x)
        x = self.emb(x)
        adjs = [self.adj1, self.adj2, self.adj3]
        for blk in self.g3d:
            x = blk(x, adjs)
        xt = self.cnx(x.mean(2).permute(0, 2, 1))   # (B, ch, T)
        xt_pool = _masked_mean(xt, mask, dim=-1)     # (B, ch) masked mean over T
        logits  = self.head(xt_pool).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  7. SkelFormer — Adaptive hierarchical temporal-split skeleton transformer
#     Tier A, Section 8: DOI 10.1371/journal.pone.0340390 (Jan 2026, PLOS ONE)
#
#     Mechanism: SKT Block with a Temporal-Split submodule that adaptively
#     divides the skeleton sequence into learned temporal segments at multiple
#     scales. Here implemented as a 2-scale version:
#       Scale 1 (coarse): T → S1 segments via learned split query
#       Scale 2 (fine):   T → S2 segments (S2 = 2 × S1)
#     Each scale: GCN across joints, then cross-attention from split queries
#     to the full sequence with key padding mask, multi-scale fusion → head.
# ═══════════════════════════════════════════════════════════════════════════════

class _SkelSplitBlock(nn.Module):
    """
    SKT Block: joint GCN + temporal split at one scale.
    Outputs a segment descriptor of shape (B, n_seg, d_model).
    """

    def __init__(self, in_ch, d_model, n_seg, n_heads=4, dropout=0.1):
        super().__init__()
        self.joint_proj = nn.Linear(in_ch, d_model)
        self.joint_pos  = nn.Parameter(torch.randn(1, N_JOINTS, d_model) * 0.02)
        # Spatial: joint-level self-attention per frame
        sp_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                              dropout=dropout, batch_first=True, norm_first=True)
        self.sp_tf = nn.TransformerEncoder(sp_layer, num_layers=1)
        # Temporal split: learnable segment queries attend to the full T sequence
        self.seg_q = nn.Parameter(torch.randn(1, n_seg, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, adj, valid_mask=None):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        # Spatial GCN: project + joint-level attention per frame
        xj = self.joint_proj(x.reshape(B * T, N, C)) + self.joint_pos
        xj = self.sp_tf(xj).reshape(B, T, N, -1).mean(2)   # (B, T, d_model)
        # Temporal split: segment queries cross-attend to the T descriptors
        q = self.seg_q.expand(B, -1, -1)                   # (B, n_seg, d_model)
        kpm = ~valid_mask if valid_mask is not None else None
        seg, _ = self.cross_attn(q, xj, xj, key_padding_mask=kpm) # (B, n_seg, d_model)
        return self.norm(seg)                               # (B, n_seg, d_model)


class SkelFormer(nn.Module):
    """
    SkelFormer with 2-scale SKT Blocks (coarse S1=10, fine S2=20) with key padding masks.
    Fuses both scales → head.
    Input: (B, T, 33, 2)
    """

    def __init__(self, in_ch=2, d_model=64, n_heads=4,
                 s1=10, s2=20, dropout=0.3):
        super().__init__()
        self.register_buffer("adj", _ADJ)
        self.block_coarse = _SkelSplitBlock(in_ch, d_model, s1, n_heads)
        self.block_fine   = _SkelSplitBlock(in_ch, d_model, s2, n_heads)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x, calibrate=False):
        mask = _valid_mask(x)
        adj = self.adj
        coarse = self.block_coarse(x, adj, valid_mask=mask).mean(1)   # (B, d_model)
        fine   = self.block_fine(x, adj, valid_mask=mask).mean(1)     # (B, d_model)
        logits = self.head(torch.cat([coarse, fine], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  8. MTC-Former — Multi-Grained Temporal Clip Transformer
#     Tier A, Section 8: MDPI Applied Sciences 2025
#
#     Segments into K clips at multiple granularities simultaneously.
#     Three branches: coarse (K=5), mid (K=10), fine (K=20).
#     Each branch: masked mean-pool within clips → Transformer → CLS token.
#     Branch descriptors fused → head.
# ═══════════════════════════════════════════════════════════════════════════════

class _ClipBranch(nn.Module):
    """One MTC-Former branch: segment into K clips → masked pool → TF encoder → CLS."""

    def __init__(self, in_dim, d_model, n_clips, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_clips = n_clips
        self.proj    = nn.Linear(in_dim, d_model)
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, n_clips + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        B, T, D = x.shape
        K = self.n_clips
        # Pad to multiple of K
        pad = (K - T % K) % K
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            if mask is not None:
                mask = F.pad(mask, (0, pad), value=False)

        if mask is not None:
            c_mask = mask.reshape(B, K, -1)
            clips = _masked_mean(x.reshape(B, K, -1, D), c_mask, dim=2)
            clip_has_data = c_mask.any(dim=-1)
        else:
            clips = x.reshape(B, K, -1, D).mean(2)           # (B, K, D)
            clip_has_data = torch.ones((B, K), dtype=torch.bool, device=x.device)

        clips = self.proj(clips)                          # (B, K, d_model)
        cls   = self.cls_tok.expand(B, -1, -1)
        tok   = torch.cat([cls, clips], 1) + self.pos_emb # (B, 1 + K, d_model)

        cls_pad_mask = torch.cat([torch.zeros((B, 1), dtype=torch.bool, device=x.device), ~clip_has_data], dim=1)
        out   = self.tf(tok, src_key_padding_mask=cls_pad_mask)
        return self.norm(out[:, 0])                       # (B, d_model) — CLS


class MTCFormer(nn.Module):
    """
    Multi-Grained Temporal Clip Transformer with clip-level valid masking.
    Input: (B, T, 33, 2)  (joint coords flattened to 66 for clip pooling)
    """

    def __init__(self, in_dim=66, d_model=64, dropout=0.3):
        super().__init__()
        self.coarse = _ClipBranch(in_dim, d_model, n_clips=5)
        self.mid    = _ClipBranch(in_dim, d_model, n_clips=10)
        self.fine   = _ClipBranch(in_dim, d_model, n_clips=20)
        self.head   = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model * 3, d_model),
            nn.GELU(), nn.Linear(d_model, 1),
        )

    def forward(self, x, calibrate=False):
        B, T, L, C = x.shape
        mask = _valid_mask(x)
        xf = x.reshape(B, T, L * C)                      # (B, T, 66)
        d = torch.cat([self.coarse(xf, mask), self.mid(xf, mask), self.fine(xf, mask)], dim=-1)
        logits = self.head(d).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  9. MTT — Multi-Scale Temporal Transformer (~2022)
#     Tier A, Section 8: segmental sampling + skeleton-TF with joint selection.
#
#     Implementation: 3-scale temporal downsampling (stride 1, 2, 4) followed by
#     a Transformer with a learned joint-importance gate (soft selection) and
#     temporal padding masks.
# ═══════════════════════════════════════════════════════════════════════════════

class MTT(nn.Module):
    """
    Multi-Scale Temporal Transformer with masked temporal pooling and attention.
    3 scales (full / stride-2 / stride-4) processed by a shared GCN-TF,
    then joint-importance gate per scale, multi-scale fusion → head.
    Input: (B, T, 33, 2)
    """

    def __init__(self, in_ch=2, d_model=64, n_heads=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.register_buffer("adj", _ADJ)
        self.joint_proj = nn.Linear(in_ch, d_model)
        self.joint_pos  = nn.Parameter(torch.randn(1, N_JOINTS, d_model) * 0.02)
        # Shared spatial encoder (joint self-attention, per frame)
        sp = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                        dropout=0.1, batch_first=True, norm_first=True)
        self.sp_tf = nn.TransformerEncoder(sp, 1)
        # Learnable joint importance gate
        self.gate = nn.Linear(d_model, 1)
        # Temporal transformer (operates on T descriptors)
        tp = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                        dropout=0.1, batch_first=True, norm_first=True)
        self.tp_tf = nn.TransformerEncoder(tp, n_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model * 3, d_model),
            nn.GELU(), nn.Linear(d_model, 1),
        )

    def _encode_scale(self, x, mask):
        """x: (B, T', 33, 2), mask: (B, T')  →  (B, d_model)"""
        B, T, N, C = x.shape
        xj = self.joint_proj(x.reshape(B * T, N, C)) + self.joint_pos
        xj = self.sp_tf(xj).reshape(B, T, N, -1)       # (B, T, N, d)
        # Soft joint gate
        gate_w = torch.softmax(self.gate(xj).squeeze(-1), dim=-1)  # (B,T,N)
        xf = (xj * gate_w.unsqueeze(-1)).sum(2)         # (B, T, d)
        tf_out = self.tp_tf(xf, src_key_padding_mask=~mask)
        out = _masked_mean(tf_out, mask, dim=1)         # (B, d)
        return out

    def forward(self, x, calibrate=False):
        # Multi-scale: full, stride-2, stride-4
        mask = _valid_mask(x)
        s1 = self._encode_scale(x, mask)
        s2 = self._encode_scale(x[:, ::2], mask[:, ::2])
        s4 = self._encode_scale(x[:, ::4], mask[:, ::4])
        logits = self.head(torch.cat([s1, s2, s4], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  10. STAR — Sparse Transformer-based Action Recognition
#      Tier A, Section 8: arXiv 2107.07089 (2021)
#
#      Mechanism: Top-K sparse self-attention across frames with zero-padded
#      frame masking and masked temporal pooling.
# ═══════════════════════════════════════════════════════════════════════════════

class _SparseAttention(nn.Module):
    """Top-K sparse multi-head self-attention with key padding mask."""

    def __init__(self, d_model, n_heads, top_k=32, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.dk = d_model // n_heads
        self.k  = top_k
        self.qkv  = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                             # (B, H, T, dk)
        scores = (q @ k.transpose(-2, -1)) / (self.dk ** 0.5)  # (B, H, T, T)

        # Mask padded key positions before top-k selection
        if mask is not None:
            key_pad = (~mask).unsqueeze(1).unsqueeze(2)     # (B, 1, 1, T)
            scores = scores.masked_fill(key_pad, float("-inf"))

        # Keep only top-K valid keys per query; mask the rest
        K = min(self.k, T)
        topk_vals, _ = scores.topk(K, dim=-1)
        thresh = topk_vals[..., -1:].detach()
        sparse_mask = scores < thresh
        scores = scores.masked_fill(sparse_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)
        out  = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class STAR(nn.Module):
    """
    Sparse Transformer for skeleton action recognition.
    Uses top-K sparse self-attention and masked temporal pooling.
    Input: (B, T, 33, 2)
    """

    def __init__(self, in_ch=2, d_model=64, n_heads=4, n_layers=3,
                 top_k=32, dropout=0.3):
        super().__init__()
        self.joint_proj = nn.Linear(in_ch, d_model)
        self.joint_pos  = nn.Parameter(torch.randn(1, N_JOINTS, d_model) * 0.02)
        self.frame_emb  = nn.Embedding(300, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "sparse_attn": _SparseAttention(d_model, n_heads, top_k),
                "norm1":       nn.LayerNorm(d_model),
                "ffn":         nn.Sequential(
                    nn.Linear(d_model, d_model * 4), nn.GELU(),
                    nn.Dropout(dropout), nn.Linear(d_model * 4, d_model),
                ),
                "norm2":       nn.LayerNorm(d_model),
            }))

        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, calibrate=False):
        B, T, N, C = x.shape
        mask = _valid_mask(x)
        # Pool joints → per-frame descriptor
        xj = self.joint_proj(x.reshape(B * T, N, C)) + self.joint_pos
        xf = xj.reshape(B, T, N, -1).mean(2)            # (B, T, d_model)
        xf = xf + self.frame_emb(torch.arange(T, device=x.device)).unsqueeze(0)

        for layer in self.layers:
            xf = layer["norm1"](xf + layer["sparse_attn"](xf, mask=mask))
            xf = layer["norm2"](xf + layer["ffn"](xf))

        xf_pool = _masked_mean(xf, mask, dim=1)
        logits = self.head(xf_pool).squeeze(-1)
        return torch.sigmoid(logits), logits


# ═══════════════════════════════════════════════════════════════════════════════
#  Scikit-learn baseline factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_sklearn_baseline(name: str, seed: int = 42):
    """
    Return a scikit-learn Pipeline. Features: mean-pooled (N, 66).
    Supported: 'lr', 'svm', 'rf', 'xgboost'
    """
    if name == "lr":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                 random_state=seed, C=1.0)
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced",
                  random_state=seed, C=1.0)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                     class_weight="balanced", random_state=seed, n_jobs=-1)
    elif name == "xgboost":
        if not _XGBOOST_AVAILABLE:
            raise ImportError("xgboost not installed.  Run: pip install xgboost")
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", random_state=seed, n_jobs=-1)
    else:
        raise ValueError(f"Unknown baseline '{name}'. Choose: lr, svm, rf, xgboost")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def extract_sklearn_features(npy_paths: list) -> np.ndarray:
    """Load .npy (300,33,2), mean-pool over valid frames → (N,66)."""
    feats = []
    for path in npy_paths:
        seq  = np.load(path).astype(np.float32)
        mask = np.any(seq != 0, axis=(1, 2))
        valid = seq[mask] if mask.any() else seq
        feats.append(valid.reshape(len(valid), -1).mean(0))
    return np.stack(feats, 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline registry (used by ablation.py)
# ═══════════════════════════════════════════════════════════════════════════════

PYTORCH_BASELINES = {
    # Protocol A5
    "lstm":           StackedLSTM,
    "conv1d_bilstm":  Conv1DBiLSTMAttn,
    "kinematic_cnn":  KinematicCNNLSTM,
    "stts":           STTS,
    "msg3d":          MSG3D,
    "msg3d_convnext": MSG3DConvNeXt,
    # Protocol Section 8 (Table 1 repositioning)
    "skelformer":     SkelFormer,
    "mtcformer":      MTCFormer,
    "mtt":            MTT,
    "star":           STAR,
}

SKLEARN_BASELINES = ["lr", "svm", "rf", "xgboost"]

ALL_BASELINE_IDS  = list(PYTORCH_BASELINES.keys()) + SKLEARN_BASELINES
