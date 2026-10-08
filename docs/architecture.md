# Architecture of the reference network (`src/model.py`)

Input: a prepared sequence `(B, T, 33, 2)` (T = 300; first detected frame at 0;
undetected frames zero). Validity `m_t = 1[sum |x_t| > 1e-4]`.

| Stage | Class | Computation | Output |
|---|---|---|---|
| Onset alignment | `align_onset` | shift so the first valid frame is frame 0 (`model.align_onset`) | `(B, T, 33, 2)` |
| Spatial encoder | `SpatialEncoder` | descriptor `[x; 10 dx; 5 d(10 dx)]` (198), differences zero across invalid frames; residual MLP with LayerNorm; multiplied by `m_t` | `(B, T, 128)` |
| Microkinetic encoder | `MicrokineticEncoder` | Conv1d with kernels 1, 3, 5 (32 channels each); GroupNorm (8 groups) with statistics over valid frames; LeakyReLU; re-zeroed at invalid frames | `(B, T, 96)` |
| Event gate | `EventSaliencyGate` | blocks of L = 15 frames; block mean over valid frames; MLP 96-48-1 score; invalid blocks scored -1e9; top M = 8 blocks in temporal order; tokens weighted by sigmoid(score) | `(B, 120, 96)`, frame indices, token mask |
| Temporal Transformer | `TemporalEventTransformer` | linear projection + sinusoidal encoding of the frame index; 1 pre-norm encoder layer (4 heads, width 96, feed-forward 192); key-padding mask; mean over valid tokens; Linear 96-128 + LayerNorm | `(B, 128)` |
| Classifier | `ASDMotionModel.classifier` | Linear 128-64, LayerNorm, GELU, dropout, Linear 64-1 | logit |
| Recalibration | `calibration.PlattScaler` | `p = sigmoid(exp(theta) * logit + b)`, fitted on validation children | probability |

223,106 trainable parameters. Ablation flags: `use_gate=False` (all frames reach
the Transformer), `event_block_size=1, event_top_m=120` (frame selection),
`use_transformer=False` (masked mean of the selected tokens),
`model.mask_padding=False`, `model.align_onset=False`.

With `mask_padding` and `align_onset` true, the logit does not depend on how many
undetected frames precede the first detection or follow the last one
(`tests/test_padding_invariance.py`); detection gaps inside a recording are not
covered.

## Comparison architectures (`src/comparison_models.py`)

All take the same input and kinematic channels, mask undetected frames in
recurrence, attention and pooling, and use the same training recipe
(`src/protocol.py`).

| Name | Structure |
|---|---|
| BiGRU | Linear 198-96 + LayerNorm; 2-layer bidirectional GRU (64 per direction) over frames up to the last detection; masked mean |
| TCN | Conv1d 198-96 (k = 5); four residual blocks (k = 3, dilations 1, 2, 4, 8, LayerNorm); masked mean |
| ST-GCN | 7 spatial-temporal graph blocks (32-128 channels, temporal kernel 9) on the MediaPipe graph with spatial partitioning |
| CTR-GCN | 7 blocks of channel-wise topology refinement graph convolution with multi-scale temporal convolution (32-128 channels) |
| ST-Transformer | joint embedding (6 to 64), 1 spatial attention layer per frame, joint mean, 2 temporal attention layers with key-padding mask |
