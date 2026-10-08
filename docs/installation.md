# Installation

PACE-ASD needs Python 3.11. Inference and the test suite run on the processor;
a CUDA graphics card shortens training.

```bash
git clone https://github.com/neuro-paradigm/PACE-ASD.git
cd PACE-ASD
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
```

PyTorch first, for your platform:

```bash
# CPU (Linux, Windows)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
# macOS (Apple silicon or Intel)
pip install torch==2.1.2 torchvision==0.16.2
# CUDA 12.1
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then the pinned dependencies and the tests:

```bash
pip install -r requirements.txt
python -m pytest tests -q
```

MediaPipe installs `opencv-contrib-python`, which provides `cv2`; do not also
install `opencv-python`, which would shadow it with a different version. On
first use with `model_complexity: 2`, MediaPipe downloads its heavy pose model
(`pose_landmark_heavy.tflite`) into its package directory.

Tested platforms: Windows 11 (Python 3.11.9, CUDA 12.1), Ubuntu 26.04 under
WSL2 (Python 3.11.17, CPU), and the Ubuntu 22.04, Windows Server 2022 and macOS 14
runners of the GitHub Actions workflow in `.github/workflows/tests.yml`.
