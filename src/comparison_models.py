"""
PACE-ASD — comparison architectures for the nested cross-validation.

Every model takes the same input as PACE-ASD, a (B, T, 33, 2) batch of
prepared pose sequences, derives the same masked kinematic channels
(position, 10 x first difference, 5 x second difference of that), and
returns (probability, logit). Frames without a detected pose are excluded
from recurrence, attention and pooling, so all models see the same
information.

  BiGRU          bidirectional gated recurrent network (Cho et al., 2014)
  TCN            residual dilated 1-D temporal convolutions (Bai et al., 2018)
  STGCN          spatial-temporal graph convolution (Yan et al., 2018)
  CTRGCN         channel-wise topology refinement GCN (Chen et al., 2021)
  STTransformer  factorised spatial-then-temporal Transformer
                 (after Plizzari et al., 2021)

Widths and depths are reduced relative to the published configurations,
which were designed for datasets with tens of thousands of sequences.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_JOINTS = 33
MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    (0, 11), (0, 12),            # head attached to the shoulders
]


# ── shared helpers ────────────────────────────────────────────────────────────

def frame_mask(x: torch.Tensor) -> torch.Tensor:
    """(B, T, J, C) -> (B, T) bool. A sequence without any valid frame is
    treated as fully valid so that attention and pooling stay defined."""
    m = x.abs().sum(dim=(-2, -1)) > 1e-4
    return m | ~m.any(dim=1, keepdim=True)


def kinematics(x: torch.Tensor) -> torch.Tensor:
    """(B, T, J, 2) -> (B, T, J, 6): position, velocity, acceleration with the
    same scaling and boundary masking as the PACE-ASD spatial encoder."""
    valid = x.abs().sum(dim=(-2, -1)) > 1e-4
    vel = torch.zeros_like(x)
    vel[:, 1:] = (x[:, 1:] - x[:, :-1]) * 10.0
    vel[:, 1:][~(valid[:, 1:] & valid[:, :-1])] = 0.0
    acc = torch.zeros_like(x)
    acc[:, 2:] = (vel[:, 2:] - vel[:, 1:-1]) * 5.0
    acc[:, 2:][~(valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2])] = 0.0
    return torch.cat([x, vel, acc], dim=-1)


def masked_mean(h: torch.Tensor, m: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean of h over `dim`, counting positions where m is True."""
    w = m.to(h.dtype)
    while w.dim() < h.dim():
        w = w.unsqueeze(-1)
    if dim != 1:
        w = w.movedim(1, dim)
    return (h * w).sum(dim=dim) / w.sum(dim=dim).clamp(min=1.0)


def _head(d_in: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_in // 2), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(d_in // 2, 1))


class DilatedTemporalConv(nn.Module):
    """Temporal convolution with 'same' padding and dilation d along the time
    axis of (B, C, T) or (B, C, T, V) tensors, computed as an ordinary
    (undilated) convolution over the d interleaved subsequences. The result
    equals the dilated convolution exactly; it avoids the slow deterministic
    cuDNN kernels for dilated convolutions."""

    def __init__(self, cin, cout, k, dilation=1, stride=1, dims=1):
        super().__init__()
        self.k, self.d, self.s, self.dims = k, dilation, stride, dims
        self.conv = (nn.Conv1d(cin, cout, k) if dims == 1 else nn.Conv2d(cin, cout, (k, 1)))

    def forward(self, x):
        d, k = self.d, self.k
        T = x.shape[2]
        pad = (k - 1) * d // 2
        tail = (0, 0) if self.dims == 2 else ()
        xp = F.pad(x, (*tail, pad, pad)) if self.dims == 2 else F.pad(x, (pad, pad))
        if d == 1:
            y = self.conv(xp)
        else:
            L = xp.shape[2]
            r = (-L) % d
            if r:
                xp = F.pad(xp, (0, 0, 0, r)) if self.dims == 2 else F.pad(xp, (0, r))
                L += r
            B, C = xp.shape[:2]
            rest = xp.shape[3:]
            xs = xp.reshape(B, C, L // d, d, *rest).movedim(3, 1).reshape(B * d, C, L // d, *rest)
            z = self.conv(xs)
            Q = z.shape[2]
            y = z.reshape(B, d, -1, Q, *rest).movedim(1, 3).reshape(B, -1, Q * d, *rest)
        y = y[:, :, :T]
        return y[:, :, ::self.s] if self.s > 1 else y


class _Base(nn.Module):
    def output(self, z):
        logits = self.head(z).squeeze(-1)
        return torch.sigmoid(logits), logits


# ── recurrent and convolutional ───────────────────────────────────────────────

class BiGRU(_Base):
    def __init__(self, hidden: int = 64, n_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(198, 96), nn.LayerNorm(96), nn.GELU())
        self.gru = nn.GRU(96, hidden, n_layers, batch_first=True, bidirectional=True,
                          dropout=0.2 if n_layers > 1 else 0.0)
        self.head = _head(2 * hidden, dropout)

    def forward(self, x, calibrate=False):
        B, T = x.shape[:2]
        m = frame_mask(x)
        h = self.inp(kinematics(x).reshape(B, T, -1))
        # recurrence runs from frame 0 to the last valid frame; internal
        # detection gaps are zero inputs
        last = torch.where(m, torch.arange(T, device=x.device), -1).max(dim=1).values
        lengths = (last + 1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True,
                                                   enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return self.output(masked_mean(out, m, 1))


class _TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.c1 = DilatedTemporalConv(ch, ch, 3, dilation)
        self.c2 = DilatedTemporalConv(ch, ch, 3, dilation)
        self.n1, self.n2 = nn.LayerNorm(ch), nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, m):                       # h: (B, C, T)
        y = self.n1(self.c1(h).transpose(1, 2)).transpose(1, 2)
        y = self.drop(F.gelu(y))
        y = self.n2(self.c2(y).transpose(1, 2)).transpose(1, 2)
        return F.gelu(h + y) * m[:, None, :].to(h.dtype)


class TCN(_Base):
    def __init__(self, ch: int = 96, dilations=(1, 2, 4, 8), dropout: float = 0.4):
        super().__init__()
        self.inp = nn.Conv1d(198, ch, 5, padding=2)
        self.blocks = nn.ModuleList([_TCNBlock(ch, d, 0.1) for d in dilations])
        self.head = _head(ch, dropout)

    def forward(self, x, calibrate=False):
        B, T = x.shape[:2]
        m = frame_mask(x)
        h = self.inp(kinematics(x).reshape(B, T, -1).transpose(1, 2))
        h = h * m[:, None, :].to(h.dtype)
        for blk in self.blocks:
            h = blk(h, m)
        return self.output(masked_mean(h.transpose(1, 2), m, 1))


# ── graph convolutional ───────────────────────────────────────────────────────

def _hop_distance(n=N_JOINTS, edges=MEDIAPIPE_EDGES, max_hop=1):
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    hop = np.full((n, n), np.inf)
    mats = [np.linalg.matrix_power(A + np.eye(n), d) for d in range(max_hop + 1)]
    for d in range(max_hop, -1, -1):
        hop[mats[d] > 0] = d
    return A, hop


def spatial_partition(center=(23, 24)) -> np.ndarray:
    """ST-GCN 'spatial' partitioning into root, centripetal and centrifugal
    subsets, relative to the nearer hip landmark; column-normalised."""
    A, hop = _hop_distance()
    n = N_JOINTS
    G = np.zeros((n, n))
    for i, j in MEDIAPIPE_EDGES:
        G[i, j] = G[j, i] = 1
    dist = np.full(n, np.inf)                       # graph distance to the hips
    for c in center:
        d = np.full(n, np.inf); d[c] = 0; frontier = [c]
        while frontier:
            nxt = []
            for u in frontier:
                for v in np.flatnonzero(G[u]):
                    if d[v] > d[u] + 1:
                        d[v] = d[u] + 1; nxt.append(v)
            frontier = nxt
        dist = np.minimum(dist, d)
    adj = (hop <= 1).astype(float)
    norm = adj / adj.sum(axis=0, keepdims=True)
    root, close, far = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if hop[j, i] <= 1:
                if dist[j] == dist[i]:
                    root[j, i] = norm[j, i]
                elif dist[j] > dist[i]:
                    far[j, i] = norm[j, i]
                else:
                    close[j, i] = norm[j, i]
    return np.stack([root, close, far]).astype(np.float32)


class _STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True, dropout=0.1):
        super().__init__()
        K = A.shape[0]
        self.register_buffer("A", A)
        self.importance = nn.Parameter(torch.ones_like(A))
        self.gcn = nn.Conv2d(cin, cout * K, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout))
        if not residual:
            self.res = None
        elif cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))
        self.stride = stride

    def forward(self, x, m):                       # x: (B, C, T, V)
        B, _, T, V = x.shape
        y = self.gcn(x).view(B, self.A.shape[0], -1, T, V)
        y = torch.einsum("bkctv,kvw->bctw", y, self.A * self.importance)
        y = self.tcn(y)
        if self.res is not None:
            y = y + self.res(x)
        m = m[:, ::self.stride]
        return F.relu(y) * m[:, None, :, None].to(y.dtype), m


class STGCN(_Base):
    def __init__(self, channels=(32, 32, 32, 64, 64, 128, 128), dropout: float = 0.4):
        super().__init__()
        A = torch.from_numpy(spatial_partition())
        self.data_bn = nn.BatchNorm1d(6 * N_JOINTS)
        layers, cin = [], 6
        for i, c in enumerate(channels):
            stride = 2 if i > 0 and c != channels[i - 1] else 1
            layers.append(_STGCNBlock(cin, c, A, stride, residual=i > 0))
            cin = c
        self.layers = nn.ModuleList(layers)
        self.head = _head(cin, dropout)

    def forward(self, x, calibrate=False):
        B, T, V, _ = x.shape
        m = frame_mask(x)
        k = kinematics(x)                                       # (B, T, V, 6)
        k = self.data_bn(k.permute(0, 2, 3, 1).reshape(B, V * 6, T))
        h = k.view(B, V, 6, T).permute(0, 2, 3, 1) * m[:, None, :, None].to(x.dtype)
        for layer in self.layers:
            h, m = layer(h, m)
        h = h.mean(dim=3).transpose(1, 2)                       # (B, T', C)
        return self.output(masked_mean(h, m, 1))


class _CTRGC(nn.Module):
    def __init__(self, cin, cout, reduction=8):
        super().__init__()
        rel = 8 if cin <= 16 else cin // reduction
        self.c1, self.c2 = nn.Conv2d(cin, rel, 1), nn.Conv2d(cin, rel, 1)
        self.c3, self.c4 = nn.Conv2d(cin, cout, 1), nn.Conv2d(rel, cout, 1)

    def forward(self, x, A, alpha, m):
        w = m[:, None, :, None].to(x.dtype)
        den = w.sum(dim=2).clamp(min=1.0)
        x1 = (self.c1(x) * w).sum(dim=2) / den                  # masked mean over T
        x2 = (self.c2(x) * w).sum(dim=2) / den
        rel = torch.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))   # (B, R, V, V)
        rel = self.c4(rel) * alpha + A[None, None]
        return torch.einsum("bcuv,bctv->bctu", rel, self.c3(x))


class _UnitGCN(nn.Module):
    def __init__(self, cin, cout, A):
        super().__init__()
        self.PA = nn.Parameter(A.clone())
        self.alpha = nn.Parameter(torch.zeros(1))
        self.convs = nn.ModuleList([_CTRGC(cin, cout) for _ in range(A.shape[0])])
        self.down = nn.Identity() if cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x, m):
        y = sum(conv(x, self.PA[i], self.alpha, m) for i, conv in enumerate(self.convs))
        return F.relu(self.bn(y) + self.down(x))


class _TemporalConv(nn.Sequential):
    def __init__(self, cin, cout, k, stride=1, dilation=1):
        if dilation == 1:
            conv = nn.Conv2d(cin, cout, (k, 1), (stride, 1), ((k - 1) // 2, 0))
        else:
            conv = DilatedTemporalConv(cin, cout, k, dilation, stride, dims=2)
        super().__init__(conv, nn.BatchNorm2d(cout))


class _MultiScaleTCN(nn.Module):
    def __init__(self, cin, cout, stride=1, dilations=(1, 2)):
        super().__init__()
        nb = len(dilations) + 2
        bc = cout // nb
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(),
                           _TemporalConv(bc, bc, 5, stride, d)) for d in dilations]
            + [nn.Sequential(nn.Conv2d(cin, bc, 1), nn.BatchNorm2d(bc), nn.ReLU(),
                             nn.MaxPool2d((3, 1), (stride, 1), (1, 0)), nn.BatchNorm2d(bc)),
               nn.Sequential(nn.Conv2d(cin, cout - bc * (nb - 1), 1, (stride, 1)),
                             nn.BatchNorm2d(cout - bc * (nb - 1)))])

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


class _CTRBlock(nn.Module):
    def __init__(self, cin, cout, A, stride=1, residual=True):
        super().__init__()
        self.gcn = _UnitGCN(cin, cout, A)
        self.tcn = _MultiScaleTCN(cout, cout, stride)
        if not residual:
            self.res = None
        elif cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = _TemporalConv(cin, cout, 1, stride)
        self.stride = stride

    def forward(self, x, m):
        y = self.tcn(self.gcn(x, m))
        if self.res is not None:
            y = y + self.res(x)
        m = m[:, ::self.stride]
        return F.relu(y) * m[:, None, :, None].to(y.dtype), m


class CTRGCN(_Base):
    def __init__(self, channels=(32, 32, 32, 64, 64, 128, 128), dropout: float = 0.4):
        super().__init__()
        A = torch.from_numpy(spatial_partition())
        self.data_bn = nn.BatchNorm1d(6 * N_JOINTS)
        layers, cin = [], 6
        for i, c in enumerate(channels):
            stride = 2 if i > 0 and c != channels[i - 1] else 1
            layers.append(_CTRBlock(cin, c, A, stride, residual=i > 0))
            cin = c
        self.layers = nn.ModuleList(layers)
        self.head = _head(cin, dropout)

    def forward(self, x, calibrate=False):
        B, T, V, _ = x.shape
        m = frame_mask(x)
        k = self.data_bn(kinematics(x).permute(0, 2, 3, 1).reshape(B, V * 6, T))
        h = k.view(B, V, 6, T).permute(0, 2, 3, 1) * m[:, None, :, None].to(x.dtype)
        for layer in self.layers:
            h, m = layer(h, m)
        return self.output(masked_mean(h.mean(dim=3).transpose(1, 2), m, 1))


# ── Transformer ───────────────────────────────────────────────────────────────

def _sinusoid(T: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class STTransformer(_Base):
    def __init__(self, d: int = 64, heads: int = 4, t_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        self.joint_emb = nn.Linear(6, d)
        self.joint_pos = nn.Parameter(torch.randn(1, N_JOINTS, d) * 0.02)
        enc = lambda: nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, activation="gelu",
                                                 batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(enc(), 1)
        self.temporal = nn.TransformerEncoder(enc(), t_layers)
        self.head = _head(d, dropout)

    def forward(self, x, calibrate=False):
        B, T, V, _ = x.shape
        m = frame_mask(x)
        k = kinematics(x)
        h = torch.zeros(B, T, self.joint_pos.shape[-1], device=x.device, dtype=x.dtype)
        # spatial attention only over frames with a detected pose
        sel = m.reshape(-1)
        tok = self.joint_emb(k.reshape(B * T, V, 6)[sel]) + self.joint_pos
        h.view(B * T, -1)[sel] = self.spatial(tok).mean(dim=1)
        h = h + _sinusoid(T, h.shape[-1], x.device)
        h = self.temporal(h, src_key_padding_mask=~m)
        return self.output(masked_mean(h, m, 1))


COMPARISON_MODELS = {
    "BiGRU": BiGRU,
    "TCN": TCN,
    "ST-GCN": STGCN,
    "CTR-GCN": CTRGCN,
    "ST-Transformer": STTransformer,
}
