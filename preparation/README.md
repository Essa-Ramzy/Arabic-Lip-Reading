# Preparation Module

## Purpose

The preparation module handles video preprocessing for the Arabic Lip Reading API, focusing on face detection, landmark extraction, and mouth region cropping. This module prepares raw video inputs for subsequent processing by the ML models.

## Folder Structure

```
preparation/
├── retinaface/
│   ├── 20words_mean_face.npy         # Mean face template for alignment
│   ├── detector.py                   # Main landmarks detector class
│   ├── mouth_cropping.py             # AVSR data loader and mouth cropping
│   ├── video_process.py              # Video preprocessing pipeline
│   └── ibug/                         # Face detection and alignment modules
│       ├── face_alignment/           # Facial landmark detection
│       │   ├── fan/                  # Face Alignment Network (FAN)
│       │   ├── utils.py              # Landmark visualization utilities
│       │   └── __init__.py
│       └── face_detection/           # Face detection components
│           ├── retina_face/          # RetinaFace detector implementation
│           ├── s3fd/                 # S3FD detector implementation
│           ├── utils/                # Detection utilities
│           ├── download_weights.py   # Weight download script
│           └── __init__.py
```

## File Descriptions

### Core Files

- **`detector.py`**: Main `LandmarksDetector` class combining face detection and landmark extraction
- **`mouth_cropping.py`**: `AVSRDataLoader` class for complete video preprocessing pipeline
- **`video_process.py`**: Low-level video processing operations
- **`20words_mean_face.npy`**: Reference mean face template for mouth region normalization

### Subdirectories

- **`ibug/`**: Third-party face detection and alignment modules (from iBUG research group)
  - **`face_alignment/`**: Facial landmark detection using Face Alignment Networks
  - **`face_detection/`**: Face detection using RetinaFace and S3FD algorithms

## Internal Usage

### Basic Video Processing

```python
from preparation.retinaface.detector import LandmarksDetector
from preparation.retinaface.video_process import VideoProcess

# Initialize components
detector = LandmarksDetector(device="cuda:0")
processor = VideoProcess(convert_gray=True)

# Process video frames
landmarks = detector(video_frames)
processed_video = processor(video_frames, landmarks)
```

### Complete Pipeline

```python
from preparation.retinaface.mouth_cropping import AVSRDataLoader

# Initialize data loader
loader = AVSRDataLoader()

# Process video file
video_tensor = loader("path/to/video.mp4")
```

## Key Components

### Face Detection

- **RetinaFace**: Primary face detection algorithm
- **S3FD**: Alternative face detection method
- Configurable confidence thresholds and model variants

### Landmark Detection

- **FAN (Face Alignment Network)**: 68-point facial landmark detection
- Support for multiple FAN variants (2DFAN2, 2DFAN4)
- GPU acceleration support

### Video Processing

- Mouth region cropping and normalization
- Grayscale conversion
- Frame-by-frame processing with landmark tracking

## Dependencies

- PyTorch
- OpenCV
- NumPy
- torchvision
- Custom iBUG face detection/alignment modules

## Notes for Contributors

- All face detection models require pre-trained weights
- GPU acceleration recommended for real-time processing
- Landmark detection expects single-face scenarios (selects largest face)
- Video preprocessing maintains temporal consistency across frames
