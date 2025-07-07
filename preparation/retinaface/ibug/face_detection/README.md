# Face Detection Module

## Purpose

The face detection module provides robust face detection capabilities using state-of-the-art algorithms. This module includes implementations of RetinaFace and S3FD (S3 Face Detector) with additional utilities for face tracking and pose estimation.

## Folder Structure

```
face_detection/
├── retina_face/                      # RetinaFace detector
│   ├── box_utils.py                  # Bounding box utilities
│   ├── config.py                     # Model configurations
│   ├── prior_box.py                  # Prior box generation
│   ├── py_cpu_nms.py                 # Non-maximum suppression
│   ├── retina_face.py                # RetinaFace model
│   ├── retina_face_net.py            # Network architecture
│   ├── retina_face_predictor.py      # Inference wrapper
│   ├── weights/                      # Pre-trained weights
│   │   ├── Resnet50_Final.pth        # ResNet50 backbone weights
│   │   └── mobilenet0.25_Final.pth   # MobileNet backbone weights
│   └── __init__.py
├── s3fd/                             # S3FD detector
│   ├── s3fd_net.py                   # S3FD network architecture
│   ├── s3fd_predictor.py             # S3FD inference wrapper
│   ├── utils.py                      # S3FD utilities
│   ├── weights/                      # Pre-trained weights
│   └── __init__.py
├── utils/                            # Detection utilities
│   ├── head_pose_estimator.py        # Head pose estimation
│   ├── simple_face_tracker.py        # Face tracking
│   ├── data/                         # Utility data files
│   └── __init__.py
├── download_weights.py               # Weight download script
└── __init__.py
```

## File Descriptions

### RetinaFace Implementation

- **`retina_face_predictor.py`**: `RetinaFacePredictor` class for face detection inference
- **`retina_face.py`**: Main RetinaFace model implementation
- **`retina_face_net.py`**: RetinaFace network architecture
- **`box_utils.py`**: Bounding box encoding/decoding utilities
- **`prior_box.py`**: Prior box generation for anchor-based detection
- **`py_cpu_nms.py`**: CPU-based non-maximum suppression
- **`config.py`**: Model configurations for different backbones

### S3FD Implementation

- **`s3fd_predictor.py`**: `S3FDPredictor` class for S3FD inference
- **`s3fd_net.py`**: S3FD network architecture
- **`utils.py`**: S3FD-specific utilities

### Utilities

- **`utils/head_pose_estimator.py`**: Head pose estimation from facial landmarks
- **`utils/simple_face_tracker.py`**: Simple face tracking across frames
- **`download_weights.py`**: Script for downloading pre-trained model weights

## Internal Usage

### RetinaFace Detection

```python
from preparation.retinaface.ibug.face_detection import RetinaFacePredictor

# Initialize with ResNet50 backbone
predictor = RetinaFacePredictor(
    device="cuda:0",
    threshold=0.8,
    model=RetinaFacePredictor.get_model("resnet50")
)

# Detect faces in image
faces = predictor(image, rgb=False)
```

### S3FD Detection

```python
from preparation.retinaface.ibug.face_detection import S3FDPredictor

# Initialize S3FD detector
predictor = S3FDPredictor(device="cuda:0", threshold=0.6)

# Detect faces
faces = predictor(image)
```

### Model Selection

```python
# Use MobileNet backbone for speed
mobile_model = RetinaFacePredictor.get_model("mobilenet0.25")
predictor = RetinaFacePredictor(device="cuda:0", model=mobile_model)
```

## Key Components

### RetinaFacePredictor Class

- **Model Management**: Automatic weight loading and model initialization
- **Inference**: Single-shot face detection with landmark prediction
- **Configuration**: Flexible backbone and threshold configuration
- **NMS**: Built-in non-maximum suppression for duplicate removal

### S3FDPredictor Class

- **Architecture**: Scale-invariant face detection
- **Multi-Scale**: Effective detection across different face sizes
- **Speed**: Optimized for real-time applications

### Detection Utilities

- **Prior Box Generation**: Anchor generation for detection
- **Box Utilities**: Bounding box encoding, decoding, and NMS
- **Pose Estimation**: Head pose estimation from facial landmarks

## Model Configurations

### RetinaFace Variants

- **ResNet50**: High accuracy, slower inference (~20-30 FPS)
- **MobileNet0.25**: Fast inference, lower accuracy (~40-60 FPS)

### Detection Parameters

- **Threshold**: Detection confidence threshold (default: 0.8)
- **NMS Threshold**: Non-maximum suppression threshold
- **Input Size**: Input image resolution for detection

### S3FD Configuration

- **Multi-Scale**: Detects faces from 16x16 to 512x512 pixels
- **Anchor-Free**: No anchor generation required
- **Fast NMS**: Optimized non-maximum suppression

## Detection Output Format

### Face Bounding Boxes

- **Format**: [x1, y1, x2, y2, confidence]
- **Coordinate System**: Pixel coordinates in input image space
- **Confidence**: Detection confidence score (0-1)

### Optional Landmarks (RetinaFace)

- **Format**: 5-point landmarks (eyes, nose, mouth corners)
- **Coordinate System**: Pixel coordinates
- **Use**: Face alignment and quality assessment

## Performance Characteristics

### RetinaFace Performance

- **ResNet50**: ~25 FPS on GTX 1080, high accuracy
- **MobileNet0.25**: ~50 FPS on GTX 1080, good accuracy
- **Memory**: ~2-4 GB GPU memory depending on input size

### S3FD Performance

- **Speed**: ~30-40 FPS on GTX 1080
- **Accuracy**: Good performance on various face sizes
- **Memory**: ~1-2 GB GPU memory

## Internal Usage Tips

### Performance Optimization

- Use GPU acceleration for real-time applications
- Adjust detection threshold based on application requirements
- Consider input resolution vs. speed trade-offs
- Batch process multiple images when possible

### Quality Assurance

- Validate detection confidence scores
- Apply temporal smoothing for video sequences
- Use face size filtering to remove false positives

### Integration Points

- Designed to work with downstream landmark detection
- Outputs compatible with face tracking utilities
- Maintains coordinate system consistency

## Dependencies

- PyTorch
- OpenCV
- NumPy
- Pre-trained model weights (downloaded automatically)

## Notes for Contributors

- Model weights are loaded automatically on first use
- GPU memory usage scales with input image size
- The module supports both single image and batch processing
- Detection accuracy depends on image quality and lighting conditions
- Consider face size and pose variations when setting thresholds
