# iBUG Face Processing Module

## Purpose

The iBUG module provides comprehensive face detection and alignment capabilities for the Arabic Lip Reading API. This module integrates third-party face processing libraries from the iBUG research group, offering robust face detection and precise facial landmark extraction.

## Folder Structure

```
ibug/
├── face_alignment/                   # Facial landmark detection
│   ├── fan/                          # Face Alignment Network models
│   │   ├── fan.py                    # FAN model architecture
│   │   ├── fan_predictor.py          # FAN inference wrapper
│   │   ├── weights/                  # Pre-trained model weights
│   │   └── __init__.py
│   ├── utils.py                      # Landmark visualization utilities
│   └── __init__.py
└── face_detection/                   # Face detection implementations
    ├── retina_face/                  # RetinaFace detector
    │   ├── box_utils.py              # Bounding box utilities
    │   ├── config.py                 # Model configurations
    │   ├── prior_box.py              # Prior box generation
    │   ├── py_cpu_nms.py             # Non-maximum suppression
    │   ├── retina_face.py            # RetinaFace model
    │   ├── retina_face_net.py        # Network architecture
    │   ├── retina_face_predictor.py  # Inference wrapper
    │   ├── weights/                  # Pre-trained weights
    │   └── __init__.py
    ├── s3fd/                         # S3FD detector
    │   ├── s3fd_net.py               # S3FD network
    │   ├── s3fd_predictor.py         # S3FD inference wrapper
    │   ├── utils.py                  # S3FD utilities
    │   ├── weights/                  # Pre-trained weights
    │   └── __init__.py
    ├── utils/                        # Detection utilities
    │   ├── head_pose_estimator.py    # Head pose estimation
    │   ├── simple_face_tracker.py    # Face tracking
    │   └── data/                     # Utility data files
    ├── download_weights.py           # Weight download script
    └── __init__.py
```

## File Descriptions

### Face Alignment Module

- **`face_alignment/fan/`**: Face Alignment Network implementation for precise facial landmark detection
- **`face_alignment/utils.py`**: Utilities for landmark visualization and connectivity mapping

### Face Detection Module

- **`face_detection/retina_face/`**: RetinaFace implementation for robust face detection
- **`face_detection/s3fd/`**: S3FD (S3 Face Detector) alternative implementation
- **`face_detection/utils/`**: Additional utilities for face tracking and pose estimation
- **`face_detection/download_weights.py`**: Script for downloading pre-trained model weights

## Internal Usage

### Face Detection with RetinaFace

```python
from preparation.retinaface.ibug.face_detection import RetinaFacePredictor

# Initialize predictor
predictor = RetinaFacePredictor(
    device="cuda:0",
    threshold=0.8,
    model=RetinaFacePredictor.get_model("resnet50")
)

# Detect faces
faces = predictor(image, rgb=False)
```

### Facial Landmark Detection

```python
from preparation.retinaface.ibug.face_alignment import FANPredictor

# Initialize landmark predictor
predictor = FANPredictor(device="cuda:0")

# Extract landmarks
landmarks, scores = predictor(image, face_boxes, rgb=True)
```

## Key Components

### RetinaFace Detection

- Multiple backbone options (ResNet50, MobileNet0.25)
- Configurable detection thresholds
- Efficient prior box generation and NMS

### Face Alignment Network (FAN)

- 68-point facial landmark detection
- Multiple model variants (2DFAN2, 2DFAN4)
- High-precision landmark localization

### S3FD Detection

- Alternative face detection method
- Optimized for speed-accuracy trade-offs
- Complementary to RetinaFace detection

## Model Configurations

### Available Models

- **RetinaFace ResNet50**: High accuracy, slower inference
- **RetinaFace MobileNet0.25**: Faster inference, lower accuracy
- **2DFAN2**: 2-module Face Alignment Network
- **2DFAN4**: 4-module Face Alignment Network (higher accuracy)

### Configuration Options

- Detection confidence thresholds
- NMS parameters
- Input image sizes
- Device selection (CPU/GPU)

## Dependencies

- PyTorch
- OpenCV
- NumPy
- Pre-trained model weights (downloaded automatically)

## Notes for Contributors

- All models require pre-trained weights for initialization
- GPU acceleration recommended for real-time applications
- The module is designed to work with single or multiple face scenarios
- Integration with upstream video processing pipeline maintained
- Model weights are cached locally after first download
