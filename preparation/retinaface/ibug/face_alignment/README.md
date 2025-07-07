# Face Alignment Module

## Purpose

The face alignment module provides precise facial landmark detection using Face Alignment Networks (FAN). This module extracts 68-point facial landmarks from detected faces, enabling accurate mouth region localization for lip reading applications.

## Folder Structure

```
face_alignment/
├── fan/                              # Face Alignment Network models
│   ├── fan.py                        # FAN model architecture
│   ├── fan_predictor.py              # FAN inference wrapper
│   ├── weights/                      # Pre-trained model weights
│   │   ├── 2dfan2.pth                # 2-module FAN weights
│   │   ├── 2dfan4.pth                # 4-module FAN weights
│   │   └── 2dfan2_alt.pth            # Alternative 2-module FAN
│   └── __init__.py
├── utils.py                          # Landmark visualization utilities
└── __init__.py
```

## File Descriptions

### Core Files

- **`fan/fan.py`**: Face Alignment Network model architecture implementation
- **`fan/fan_predictor.py`**: `FANPredictor` class for facial landmark detection inference
- **`utils.py`**: Utilities for landmark visualization and connectivity mapping

### Model Weights

- **`2dfan2.pth`**: 2-module FAN model (faster inference)
- **`2dfan4.pth`**: 4-module FAN model (higher accuracy)
- **`2dfan2_alt.pth`**: Alternative 2-module FAN configuration

## Internal Usage

### Basic Landmark Detection

```python
from preparation.retinaface.ibug.face_alignment import FANPredictor

# Initialize predictor
predictor = FANPredictor(device="cuda:0")

# Extract landmarks from detected faces
landmarks, scores = predictor(image, detected_faces, rgb=True)
```

### Model Selection

```python
# Use different FAN models
model_2dfan2 = FANPredictor.get_model("2dfan2")
model_2dfan4 = FANPredictor.get_model("2dfan4")

predictor = FANPredictor(device="cuda:0", model=model_2dfan4)
```

### Landmark Visualization

```python
from preparation.retinaface.ibug.face_alignment.utils import plot_landmarks

# Plot landmarks on image
plot_landmarks(image, landmarks, line_colour=(0, 255, 0), pts_colour=(0, 0, 255))
```

## Key Components

### FANPredictor Class

- **Initialization**: Configurable device and model selection
- **Inference**: Processes face crops to extract landmarks
- **Model Loading**: Automatic weight loading and JIT compilation support

### Face Alignment Network (FAN)

- **Architecture**: Hourglass-based network for landmark detection
- **Output**: 68-point facial landmarks with confidence scores
- **Variants**: 2-module and 4-module configurations

### Landmark Utilities

- **Connectivity**: Predefined landmark connections for 68-point and 100-point models
- **Visualization**: Plotting functions for landmark overlay on images
- **Filtering**: Confidence-based landmark filtering

## Model Configurations

### Available Models

- **2DFAN2**: 2-module network (faster, suitable for real-time)
- **2DFAN4**: 4-module network (higher accuracy, slower)
- **2DFAN2_ALT**: Alternative 2-module configuration

### Configuration Parameters

- **`crop_ratio`**: Face crop ratio for landmark detection (default: 0.55)
- **`input_size`**: Input image size (default: 256x256)
- **`num_modules`**: Number of hourglass modules (2 or 4)
- **`num_landmarks`**: Number of landmarks to detect (68 or 100)

## Landmark Format

### 68-Point Landmarks

- **Jaw line**: Points 0-16
- **Eyebrows**: Points 17-26
- **Nose**: Points 27-35
- **Eyes**: Points 36-47
- **Mouth**: Points 48-67

### Coordinate System

- Origin at top-left corner
- X-axis: left to right
- Y-axis: top to bottom
- Coordinates in pixel space

## Internal Usage Tips

### Performance Optimization

- Use GPU acceleration for real-time applications
- Consider JIT compilation for repeated inference
- Batch process multiple faces when possible

### Quality Assurance

- Filter landmarks based on confidence scores
- Validate landmark positions for outliers
- Use temporal smoothing for video sequences

### Integration

- Designed to work with upstream face detection
- Outputs compatible with downstream mouth cropping
- Maintains consistency with video processing pipeline

## Dependencies

- PyTorch
- OpenCV
- NumPy
- Pre-trained model weights (loaded automatically)

## Notes for Contributors

- Model weights are loaded automatically on first use
- GPU memory usage scales with input batch size
- The module supports both single image and batch processing
- Landmark accuracy depends on face detection quality
- Consider face alignment preprocessing for optimal results
