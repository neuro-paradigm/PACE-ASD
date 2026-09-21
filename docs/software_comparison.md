# PACE-ASD — Software Comparison

> **Note:** This comparison is based on publicly available documentation and source code.
> Capabilities are only listed as present (✓) when verifiable from the respective repository or paper.
> A `–` indicates the feature is absent or not documented for that tool.

## Comparison Matrix

| Capability | PACE-ASD | OpenPose [1] | MMPose [2] | SMILEchild [3] | SkelFormer [4] | MS-G3D [5] |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Input** | | | | | | |
| Markerless monocular RGB input | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Automatic pose extraction | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| No depth sensor required | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Kinematic Analysis** | | | | | | |
| 2D skeleton position trajectories | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| Velocity computation | ✓ | – | – | – | – | – |
| Acceleration computation | ✓ | – | – | – | – | – |
| Multi-scale temporal convolution | ✓ | – | ✓ | – | ✓ | ✓ |
| **Temporal Event Analysis** | | | | | | |
| Contiguous temporal block selection | ✓ | – | – | – | – | – |
| Learned block saliency scoring | ✓ | – | – | – | – | – |
| Temporal Transformer encoder | ✓ | – | – | – | ✓ | ✓ |
| **ASD Screening** | | | | | | |
| ASD screening research output | ✓ | – | – | ✓ | ✓ | ✓ |
| Calibrated probability output | ✓ | – | – | – | – | – |
| **Interpretability** | | | | | | |
| Temporal attention/evidence | ✓ | – | – | – | ✓ | ✓ |
| Body-region attribution | ✓ | – | – | – | ✓ | ✓ |
| Kinematic-stream attribution | ✓ | – | – | – | – | ✓ |
| Gate-attention coherence audit | ✓ | – | – | – | – | – |
| **Software Quality** | | | | | | |
| Open-source availability | ✓ | ✓ | ✓ | ✓* | ✓ | ✓ |
| Reproducible evaluation | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| Pre-extracted feature inference | ✓ | – | ✓ | – | ✓ | ✓ |
| Published unit tests | ✓ | ✓ | ✓ | – | ✓ | ✓ |
| Calibrated screening output | ✓ | – | – | – | – | – |

*SMILEchild availability subject to access request.

## References

[1] Cao, Z. et al. (2019). OpenPose: Realtime multi-person 2D pose estimation using part affinity fields. *TPAMI*.

[2] MMPose Contributors (2020). OpenMMLab Pose Estimation Toolbox and Benchmark. GitHub.

[3] Zunino, A. et al. (2018). Predicting autism spectrum disorder using a computer vision tool. *Pattern Recognition*.

[4] Yan, S. et al. (2026). SkelFormer. (See manuscript for full citation.)

[5] Liu, Z. et al. (2020). Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition. *CVPR*.

## Notes

- **Velocity/acceleration** in PACE-ASD: computed at runtime inside `SpatialEncoder` as finite differences with boundary protection (vel ×10, acc ×5 scaling to match position range). These are features for the model, not standalone analysis outputs — they are also exposed in `kinematics.npz` for external analysis.
- **Calibrated probability:** PACE-ASD applies Platt scaling (temperature + bias) fitted on out-of-fold validation logits, reducing Expected Calibration Error.
- **Block-saliency coherence:** PACE-ASD includes a gate-vs-attention cross-check quantifying whether Block-ESG gate scores and Transformer self-attention prioritize the same temporal events (A1: r=0.822).
