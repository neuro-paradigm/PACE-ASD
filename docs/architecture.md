# PACE-ASD — Architecture Documentation

## Overview

PACE-ASD is a lightweight Transformer-based architecture designed for markerless ASD motor screening from monocular video. The key architectural innovation is the **Block-Level Event Saliency Gate (Block-ESG)**, which selects the most kinematically informative contiguous temporal segments before applying self-attention.

## Processing Pipeline

```
Raw Video (.mp4 / .avi)
        │
        ▼  MediaPipe Pose (model_complexity=2)
2D Landmark Sequences: (T_actual, 33, 2)
        │
        ▼  Hip-centering + Inter-shoulder scale normalisation
Normalized Sequences: (T_actual, 33, 2)
        │
        ▼  Pad/truncate to T_MAX=300
Skeletal Sequence: (B, 300, 33, 2)
        │
┌────────▼──────────────────────────────────────────────────────────────────────────────────────
│  SpatialEncoder (per-frame residual MLP + LayerNorm)                       │
│  Input: (B, T, 33, 2)  concatenates [pos | vel | acc] -> D_c=198           │
│  Output: (B, T, spatial_dim=128)                                           │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
        │
┌────────▼──────────────────────────────────────────────────────────────────────────────────────
│  MicrokineticEncoder (parallel Conv1D: k=1, k=3, k=5 with GroupNorm)       │
│  Input: (B, T, 128)                                                        │
│  Output: (B, T, 3×conv1d_ch=96)  [multi-scale temporal patterns]          │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
        │
┌────────▼──────────────────────────────────────────────────────────────────────────────────────
│  Block-ESG: EventSaliencyGate (A1, A3 only; skipped in A2)                 │
│  Groups T=300 frames into N=20 blocks of L=15 frames each                 │
│  Scores each block with a 2-layer MLP gate (96→48→1)                    │
│  Selects top M=8 blocks by saliency score                                 │
│  Output: (B, M×L=120, 96) + block saliency scores (B, N=20)               │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
        │
┌────────▼──────────────────────────────────────────────────────────────────────────────────────
│  TemporalEventTransformer (A1, A2, A3 only; replaced by linear in A4)      │
│  Sinusoidal positional encoding by original frame index                   │
│  1-layer Transformer Encoder, 4 heads, d_ff = 2×input_dim                 │
│  Mean pooling → Linear → LayerNorm                                        │
│  Output: (B, spatial_dim=128)                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
        │
┌────────▼──────────────────────────────────────────────────────────────────────────────────────
│  Classifier Head: Linear(128→64) + LayerNorm + GELU + Linear(64→1)       │
│  Output: logit (B,)  →  sigmoid  →  raw P(ASD)                           │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
        │
┌────────▼──────────────────────────────────────────────────────────────────────────────────────
│  Platt Scaling: logit_cal = exp(log_T) * logit + bias (fitted on val set)  │
│  Output: calibrated P(ASD) ∈ [0, 1]                                       │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

## Module Reference

| Class | File | Description |
|---|---|---|
| `SpatialEncoder` | `src/model.py` | Per-frame MLP with LayerNorm. Computes pos+vel+acc at runtime. |
| `MicrokineticEncoder` | `src/model.py` | Parallel Conv1D (k=1,3,5) with GroupNorm for multi-scale temporal patterns. |
| `EventSaliencyGate` | `src/model.py` | Block-ESG: groups frames into blocks, scores with MLP gate, selects top-M. |
| `TemporalEventTransformer` | `src/model.py` | 1-layer 4-head Transformer with sinusoidal position encoding and mean pooling. |
| `ASDMotionModel` | `src/model.py` | Full pipeline with ablation flags (use_gate, use_transformer). |
| `PlattScaler` | `src/calibration.py` | Positive temperature + bias calibration via LBFGS. |

## Ablation Variants

| ID | use_gate | use_transformer | event_block_size | event_top_m | Description |
|---|---|---|---|---|---|
| A1 | True | True | 15 | 8 | Full PACE-ASD |
| A2 | False | True | — | — | No Block-ESG; dense attention |
| A3 | True | True | 1 | 120 | Frame-granularity gate |
| A4 | True | False | 15 | 8 | No Transformer; linear head |

## Key Design Decisions

1. **Block granularity (L=15, ~500ms):** Chosen to correspond to atomic motor primitives. Allows interpretable event selection while maintaining temporal context.
2. **GroupNorm over BatchNorm:** Avoids batch-size-dependent statistics on small clinical cohorts.
3. **LayerNorm in SpatialEncoder:** Eliminates inter-epoch normalisation shift.
4. **Sequence Mixup (35%, β=0.2):** Reduces overfitting on small N without distorting temporal structure.
5. **Sinusoidal (not learned) positional encoding:** Preserves absolute temporal position information for sparse selected events.
