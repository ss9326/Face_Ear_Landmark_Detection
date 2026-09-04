# Image demo setup

The upstream repository does **not** contain its datasets or any of the `.h5`
files referenced by the demo scripts. A model must first be obtained from the
author or trained using `train_face.py` followed by `train_face2ear.py`.

## Windows CPU environment

From PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
```

Place `saved_model_openpose_face2ear_v1.h5` in the repository root, then run:

```powershell
.\.venv\Scripts\python.exe demo_image.py path\to\face.jpg
```

The command writes `ear_landmarks.jpg` and `ear_landmarks.json`. The JSON file
contains the ear bounding box and center as well as all 55 landmark positions.

## Training recommendation

Use WSL2/Ubuntu for GPU training. Native Windows TensorFlow versions after 2.10
do not support NVIDIA GPU execution. This machine's Ubuntu 24.04 distro is
running under WSL2 and can see its NVIDIA GPU. The `wsl --status` warning refers
only to WSL1 being unavailable and does not prevent WSL2 GPU use.

Training also requires separately downloading and arranging both datasets:
300W-LP/AFW for the face teacher model and ibug-ears Collection A for the ear
model. The repository does not provide a download/preparation script, so inspect
the paths expected near the bottom of each `train_*.py` before starting.

## WSL2 GPU environment

The WSL-native checkout is at:

```bash
cd ~/projects/Face_Ear_Landmark_Detection
source ./activate-wsl.sh
```

The activation helper activates `.venv` and exposes the CUDA libraries installed
by `tensorflow[and-cuda]`. Verify the environment with:

```bash
python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

To rebuild this environment later:

```bash
python3 -m venv --clear .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-wsl.txt
source ./activate-wsl.sh
```

The RTX 5070 Ti has CUDA compute capability 12.0a. TensorFlow 2.21 detects it,
but reports that kernels will be JIT-compiled from PTX. The first real GPU model
operation can consequently take much longer than later runs.

## Exporting for Android

Export the final 55-channel `s6` ear heatmap head with a fixed batch-one input:

```bash
source ./activate-wsl.sh
python export_tflite.py --quantization float32
python export_tflite.py --quantization float16
python export_tflite.py --quantization int8 --representative-samples 100
```

All variants accept RGB float32 input shaped `[1, 368, 368, 3]`, normalized to
`[0, 1]`. The int8 variant uses integer internals but deliberately retains
float input/output so Android preprocessing and heatmap decoding are identical.
Each export writes a neighboring JSON file containing its exact tensor contract.

Validate numerical and landmark agreement with Keras before copying a model to
Android:

```bash
python validate_tflite.py --tflite saved_model/ear_landmarks_float16.tflite
```

Use `--limit 5` for a quick validation. The validator uses the same
ground-truth-guided holdout preprocessing as `evaluate_ear.py`; it measures
conversion fidelity, not automatic ear ROI detection.

## MediaPipe ROI inference

Run the Android-like pipeline on a photo:

```bash
python demo_mediapipe_roi.py my_photo.jpg \
  --output results/my_photo_mp_ear.jpg \
  --roi-output results/my_photo_mp_roi.jpg \
  --json results/my_photo_mp_ear.json
```

The script downloads the official MediaPipe Face Landmarker asset on first use.
It pads both sides of the MediaPipe face box and adds yaw-proportional padding
to the exposed-ear side. Tune this behavior with `--horizontal-padding`,
`--yaw-padding`, `--top-padding`, and `--bottom-padding`. Ear points below
`--confidence-threshold` (default `0.25`) are omitted.
