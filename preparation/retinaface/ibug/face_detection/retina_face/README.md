# RetinaFace Detection Module

## Purpose
The RetinaFace module implements the RetinaFace face detection algorithm, providing robust face detection with optional facial landmark prediction. This module supports multiple backbone architectures and is optimized for both accuracy and speed.

## Folder Structure
```
retina_face/
├── box_utils.py                      # Bounding box utilities
├── config.py                         # Model configurations
├── prior_box.py                      # Prior box generation
├── py_cpu_nms.py                     # Non-maximum suppression
├── retina_face.py                    # RetinaFace model
├── retina_face_net.py                # Network architecture
├── retina_face_predictor.py          # Inference wrapper
├── weights/                          # Pre-trained weights
│   ├── Resnet50_Final.pth            # ResNet50 backbone weights
│   └── mobilenet0.25_Final.pth       # MobileNet backbone weights
└── __init__.py
```

## File Descriptions

### Core Implementation
- **`retina_face_predictor.py`**: `RetinaFacePredictor` class for face detection inference
- **`retina_face.py`**: Main RetinaFace model implementation with classification, regression, and landmark heads
- **`retina_face_net.py`**: Network architectures including MobileNetV1, FPN, and SSH modules

### Detection Utilities
- **`box_utils.py`**: Bounding box encoding/decoding, IoU calculation, and matching utilities
- **`prior_box.py`**: Prior box generation for anchor-based detection
- **`py_cpu_nms.py`**: CPU-based non-maximum suppression implementation
- **`config.py`**: Model configurations for ResNet50 and MobileNet0.25 backbones

### Pre-trained Weights
- **`Resnet50_Final.pth`**: ResNet50 backbone weights (high accuracy)
- **`mobilenet0.25_Final.pth`**: MobileNet0.25 backbone weights (fast inference)

## Internal Usage

### Basic Face Detection
```python
from preparation.retinaface.ibug.face_detection.retina_face import RetinaFacePredictor

# Initialize with ResNet50 backbone
predictor = RetinaFacePredictor(
    device="cuda:0",
    threshold=0.8,
    model=RetinaFacePredictor.get_model("resnet50")
)

# Detect faces in image
faces = predictor(image, rgb=False)
```

### Model Selection
```python
# Use MobileNet for faster inference
mobile_model = RetinaFacePredictor.get_model("mobilenet0.25")
predictor = RetinaFacePredictor(device="cuda:0", model=mobile_model)

# Custom configuration
config = RetinaFacePredictor.create_config(
    conf_thresh=0.02,
    nms_thresh=0.4,
    top_k=750
)
predictor = RetinaFacePredictor(device="cuda:0", config=config)
```

## Key Components

### RetinaFacePredictor Class
- **Model Management**: Automatic weight loading and initialization
- **Inference**: Single-shot face detection with landmark prediction
- **Configuration**: Flexible backbone and parameter configuration
- **NMS**: Built-in non-maximum suppression for duplicate removal

### RetinaFace Model
- **Multi-Scale Detection**: FPN-based feature pyramid for different face sizes
- **Classification Head**: Face/background classification
- **Regression Head**: Bounding box regression
- **Landmark Head**: 5-point facial landmark prediction

### Network Architecture
- **Backbone**: ResNet50 or MobileNetV1 feature extractor
- **FPN**: Feature Pyramid Network for multi-scale features
- **SSH**: Single Stage Headless Face Detector modules

## Model Configurations

### ResNet50 Configuration
- **Backbone**: ResNet50
- **Input Size**: 840x840 (configurable)
- **Accuracy**: High detection accuracy
- **Speed**: ~20-30 FPS on GPU
- **Memory**: ~4 GB GPU memory

### MobileNet0.25 Configuration
- **Backbone**: MobileNetV1 (0.25 width multiplier)
- **Input Size**: 640x640 (configurable)
- **Accuracy**: Good detection accuracy
- **Speed**: ~40-60 FPS on GPU
- **Memory**: ~2 GB GPU memory

### Detection Parameters
- **`conf_thresh`**: Confidence threshold (default: 0.02)
- **`nms_thresh`**: NMS threshold (default: 0.4)
- **`top_k`**: Top K detections before NMS (default: 750)
- **`nms_top_k`**: Top K detections after NMS (default: 5000)

## Prior Box Generation

### Anchor Configuration
- **Min Sizes**: [[16, 32], [64, 128], [256, 512]]
- **Steps**: [8, 16, 32] (feature map strides)
- **Variance**: [0.1, 0.2] (encoding variance)

### Multi-Scale Detection
- **Small Faces**: 16x16 to 32x32 pixels
- **Medium Faces**: 64x64 to 128x128 pixels
- **Large Faces**: 256x256 to 512x512 pixels

## Detection Output Format

### Face Bounding Boxes
- **Format**: [x1, y1, x2, y2, confidence]
- **Coordinate System**: Pixel coordinates in input image space
- **Confidence**: Detection confidence score (0-1)

### Optional Landmarks
- **Format**: 5-point landmarks (left eye, right eye, nose, left mouth, right mouth)
- **Coordinate System**: Pixel coordinates
- **Use**: Face alignment and quality assessment

## Performance Characteristics

### Speed Benchmarks
- **ResNet50**: ~25 FPS on GTX 1080
- **MobileNet0.25**: ~50 FPS on GTX 1080
- **CPU Mode**: ~2-5 FPS depending on backbone

### Accuracy Metrics
- **ResNet50**: Superior accuracy on challenging faces
- **MobileNet0.25**: Good accuracy with speed advantage
- **Robustness**: Handles various face sizes and orientations

## Internal Usage Tips

### Performance Optimization
- Use GPU acceleration for real-time applications
- Adjust confidence threshold based on application requirements
- Consider input resolution vs. speed trade-offs
- Use MobileNet backbone for speed-critical applications

### Quality Assurance
- Validate detection confidence scores
- Apply temporal smoothing for video sequences
- Use face size filtering to remove false positives
- Consider landmark quality for face alignment

### Integration Points
- Designed to work with downstream landmark detection
- Outputs compatible with face tracking utilities
- Maintains coordinate system consistency
- Supports batch processing for multiple images

## Dependencies
- PyTorch
- OpenCV
- NumPy
- Pre-trained model weights (loaded automatically)

## Notes for Contributors
- Model weights are loaded automatically on first use
- GPU memory usage scales with input image size and batch size
- The module supports both single image and batch processing
- Detection accuracy depends on image quality and lighting conditions
- Prior box generation is cached for efficiency
