# PACE-ASD and related tools

From each project's documentation and repository (accessed October 2026).
"Not in scope" means the task lies outside the tool's documented purpose, not
that the tool could not be used for it.

| Tool | Purpose | Input -> output | Participant-level classification with nested evaluation | License |
|---|---|---|---|---|
| MediaPipe Pose | pose estimation | image or video -> 33 body landmarks | not in scope | Apache-2.0 |
| OpenPose | pose estimation | image or video -> body, hand and face keypoints | not in scope | non-commercial |
| DeepLabCut | trainable pose estimation | video -> user-defined keypoints | not in scope | LGPL-3.0 |
| OpenCap | movement dynamics from smartphones | two or more videos -> 3D kinematics and dynamics | not in scope | Apache-2.0 |
| Pose2Sim | multi-camera markerless kinematics | calibrated videos -> 3D kinematics | not in scope | BSD-3-Clause |
| PYSKL | skeleton action recognition | skeleton sequences -> action classes | fixed benchmark splits | Apache-2.0 |
| MMAction2 | video and skeleton action recognition | video or skeletons -> action classes | fixed benchmark splits | Apache-2.0 |
| **PACE-ASD** | participant-level modeling of short movement recordings | video -> landmarks -> out-of-fold probabilities, selected segments, attributions | repeated nested subject-level cross-validation, stored predictions, recording and demographic controls | Apache-2.0 |

PACE-ASD uses MediaPipe Pose for landmarks. Its comparison architectures
(ST-GCN, CTR-GCN, a spatial-temporal Transformer, BiGRU, TCN) are compact
re-implementations in `src/comparison_models.py` that share PACE-ASD's input,
masking and training recipe, so that differences between them reflect the
architecture; they are not the reference implementations of PYSKL or MMAction2.
