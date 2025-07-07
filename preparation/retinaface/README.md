# RetinaFace Video Preprocessing

## Purpose
The RetinaFace module provides comprehensive video preprocessing for the Arabic Lip Reading API, including face detection, landmark extraction, and mouth region cropping. This module serves as the primary preprocessing pipeline for converting raw video inputs into normalized mouth regions suitable for lip reading models.

## Folder Structure
```
retinaface/
├── 20words_mean_face.npy                # Mean face template for alignment
├── detector.py                          # LandmarksDetector class
├── mouth_cropping.py                    # AVSRDataLoader and processing pipeline
├── video_process.py                     # VideoProcess class for frame operations
├── README.md                            # This file
└── ibug/                                # Face detection and alignment modules
    ├── face_alignment/                  # Facial landmark detection
    │   ├── fan/                         # Face Alignment Network models
    │   │   ├── fan.py                   # FAN model architecture
    │   │   ├── fan_predictor.py         # FAN inference wrapper
    │   │   ├── weights/                 # Pre-trained model weights
    │   │   └── __init__.py
    │   ├── utils.py                     # Landmark visualization utilities
    │   └── __init__.py
    └── face_detection/                  # Face detection implementations
        ├── retina_face/                 # RetinaFace detector
        │   ├── box_utils.py             # Bounding box utilities
        │   ├── config.py                # Model configurations
        │   ├── prior_box.py             # Prior box generation
        │   ├── py_cpu_nms.py            # Non-maximum suppression
        │   ├── retina_face.py           # RetinaFace model
        │   ├── retina_face_net.py       # Network architecture
        │   ├── retina_face_predictor.py # Inference wrapper
        │   ├── weights/                 # Pre-trained weights
        │   └── __init__.py
        ├── s3fd/                        # S3FD detector
        │   ├── s3fd_net.py              # S3FD network
        │   ├── s3fd_predictor.py        # S3FD inference wrapper
        │   ├── utils.py                 # S3FD utilities
        │   ├── weights/                 # Pre-trained weights
        │   └── __init__.py
        ├── utils/                       # Detection utilities
        │   ├── head_pose_estimator.py   # Head pose estimation
        │   ├── simple_face_tracker.py   # Face tracking
        │   └── data/                    # Utility data files
        ├── download_weights.py          # Weight download script
        └── __init__.py
```

## File Descriptions

### Core Processing Files
- **`detector.py`**: Main `LandmarksDetector` class that combines RetinaFace detection with FAN landmark extraction
- **`mouth_cropping.py`**: Complete `AVSRDataLoader` pipeline for video preprocessing with mouth region extraction
- **`video_process.py`**: `VideoProcess` class for frame-level operations and mouth cropping
- **`20words_mean_face.npy`**: Reference mean face template for mouth region normalization

### iBUG Face Detection Module
- **`ibug/face_detection/`**: Face detection implementations (RetinaFace, S3FD)
- **`ibug/face_alignment/`**: Facial landmark detection using Face Alignment Networks
- **`ibug/face_detection/utils/`**: Additional utilities for face tracking and pose estimation

## Internal Usage

### Basic Landmark Detection
```python
from preparation.retinaface.detector import LandmarksDetector

# Initialize detector
detector = LandmarksDetector(device="cuda:0", model_name="resnet50")

# Process video frames
landmarks = detector(video_frames)
```

### Complete Video Preprocessing Pipeline
```python
from preparation.retinaface.mouth_cropping import AVSRDataLoader

# Initialize data loader
loader = AVSRDataLoader()

# Process video file
video_tensor = loader("path/to/video.mp4")
```

### Manual Video Processing
```python
from preparation.retinaface.video_process import VideoProcess

# Initialize video processor
processor = VideoProcess(convert_gray=True)

# Process with landmarks
processed_video = processor(video_frames, landmarks)
```

## Key Components

### LandmarksDetector
- Combines RetinaFace detection with FAN landmark extraction
- Automatically selects the largest face in multi-face scenarios
- Configurable device and model selection

### AVSRDataLoader
- Complete preprocessing pipeline from raw video to mouth crops
- Handles video loading, landmark detection, and mouth extraction
- Returns PyTorch tensors ready for model input

### VideoProcess
- Frame-level mouth cropping and normalization
- Grayscale conversion and standardization
- Temporal consistency preservation

## Model Configurations

### RetinaFace Models
- **ResNet50**: Higher accuracy, slower inference
- **MobileNet0.25**: Faster inference, lower accuracy
- Configurable detection threshold (default: 0.8)

### FAN Models
- **2DFAN2**: 2-module Face Alignment Network
- **2DFAN4**: 4-module Face Alignment Network (higher accuracy)
- 68-point facial landmark detection

## Dependencies and Installation

### Required Packages
```bash
pip install torch torchvision opencv-python numpy
```

### iBUG Modules Installation

#### Option 1: From GitHub (Recommended)
```bash
# Install Git LFS for large weight files
git lfs install

# Install ibug.face_detection
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..

# Install ibug.face_alignment
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```

#### Option 2: From Compressed Files
```bash
# Download and install ibug.face_detection
wget https://www.doc.ic.ac.uk/~pm4115/tracker/face_detection.zip -O ./face_detection.zip
unzip -o ./face_detection.zip -d ./
cd face_detection
pip install -e .
cd ..

# Download and install ibug.face_alignment
wget https://www.doc.ic.ac.uk/~pm4115/tracker/face_alignment.zip -O ./face_alignment.zip
unzip -o ./face_alignment.zip -d ./
cd face_alignment
pip install -e .
cd ..
```

## Notes for Contributors
- Pre-trained weights are required for all models (handled by installation scripts)
- GPU acceleration highly recommended for real-time processing
- The pipeline expects single-speaker scenarios (selects largest detected face)
- Video preprocessing maintains temporal consistency across frames
- All processing outputs are standardized for downstream model compatibility