# Face Alignment Network (FAN) Module

## Purpose
The FAN module implements Face Alignment Networks for precise facial landmark detection. This module provides the core neural network architecture and inference wrapper for extracting 68-point facial landmarks from face crops.

## Folder Structure
```
fan/
├── fan.py                            # FAN model architecture
├── fan_predictor.py                  # FANPredictor inference wrapper
├── weights/                          # Pre-trained model weights
│   ├── 2dfan2.pth                    # 2-module FAN weights
│   ├── 2dfan4.pth                    # 4-module FAN weights
│   └── 2dfan2_alt.pth                # Alternative 2-module FAN
└── __init__.py
```

## File Descriptions

### Core Implementation
- **`fan.py`**: Complete FAN model architecture with hourglass modules
- **`fan_predictor.py`**: `FANPredictor` class for model inference and management

### Model Weights
- **`2dfan2.pth`**: 2-module FAN model (optimized for speed)
- **`2dfan4.pth`**: 4-module FAN model (optimized for accuracy)
- **`2dfan2_alt.pth`**: Alternative 2-module FAN configuration

## Internal Usage

### Basic Landmark Detection
```python
from preparation.retinaface.ibug.face_alignment.fan import FANPredictor

# Initialize with default 2DFAN2 model
predictor = FANPredictor(device="cuda:0")

# Extract landmarks from face regions
landmarks, scores = predictor(image, face_bboxes, rgb=True)
```

### Model Selection
```python
# Use high-accuracy 4-module model
model_4fan = FANPredictor.get_model("2dfan4")
predictor = FANPredictor(device="cuda:0", model=model_4fan)

# Use alternative 2-module configuration
model_alt = FANPredictor.get_model("2dfan2_alt")
predictor = FANPredictor(device="cuda:0", model=model_alt)
```

### Configuration Options
```python
# Custom configuration
config = FANPredictor.create_config(
    crop_ratio=0.6,
    input_size=256,
    use_jit=True
)
predictor = FANPredictor(device="cuda:0", config=config)
```

## Key Components

### FANPredictor Class
- **Model Management**: Automatic weight loading and model initialization
- **Inference**: Batch processing of face crops for landmark detection
- **Configuration**: Flexible model and processing configuration
- **JIT Support**: Optional JIT compilation for optimized inference

### FAN Architecture
- **Hourglass Modules**: Encoder-decoder structure for spatial feature learning
- **Skip Connections**: Preserve fine-grained spatial information
- **Heatmap Output**: Landmark locations as spatial heatmaps
- **Multi-Scale**: Multiple hourglass modules for improved accuracy

## Model Configurations

### 2DFAN2 (Default)
- **Modules**: 2 hourglass modules
- **Speed**: Faster inference (~30-50 FPS)
- **Accuracy**: Good landmark detection accuracy
- **Use Case**: Real-time applications

### 2DFAN4
- **Modules**: 4 hourglass modules
- **Speed**: Slower inference (~15-30 FPS)
- **Accuracy**: Higher landmark detection accuracy
- **Use Case**: Offline processing, high-quality requirements

### 2DFAN2_ALT
- **Modules**: 2 hourglass modules (alternative configuration)
- **Speed**: Similar to 2DFAN2
- **Accuracy**: Different training/architecture variant
- **Use Case**: Comparative evaluation

## Configuration Parameters

### Model Parameters
- **`crop_ratio`**: Face crop ratio for landmark detection (default: 0.55)
- **`input_size`**: Input image resolution (default: 256x256)
- **`num_modules`**: Number of hourglass modules (2 or 4)
- **`hg_num_features`**: Hourglass feature channels (default: 256)
- **`hg_depth`**: Hourglass depth (default: 4)

### Processing Parameters
- **`use_jit`**: Enable JIT compilation (default: False)
- **`use_avg_pool`**: Use average pooling (model-dependent)
- **`use_instance_norm`**: Use instance normalization (default: False)

## Landmark Output Format

### 68-Point Landmarks
- **Shape**: (68, 2) array of (x, y) coordinates
- **Coordinate System**: Pixel coordinates in input image space
- **Confidence Scores**: Per-landmark confidence values
- **Ordering**: Standard 68-point face annotation format

### Landmark Regions
- **Jaw**: Points 0-16 (outline)
- **Eyebrows**: Points 17-26
- **Nose**: Points 27-35
- **Eyes**: Points 36-47
- **Mouth**: Points 48-67

## Performance Characteristics

### Speed Comparison
- **2DFAN2**: ~40 FPS on GTX 1080
- **2DFAN4**: ~20 FPS on GTX 1080
- **CPU Mode**: ~2-5 FPS depending on model

### Accuracy Metrics
- **2DFAN2**: Good landmark accuracy for most applications
- **2DFAN4**: Superior accuracy, especially for challenging poses
- **Robustness**: Handles various lighting and pose conditions

## Internal Usage Tips

### Performance Optimization
- Use GPU acceleration for real-time applications
- Enable JIT compilation for repeated inference
- Batch process multiple faces when possible
- Consider input resolution vs. accuracy trade-offs

### Quality Assurance
- Validate landmark positions for outliers
- Use confidence scores to filter unreliable detections
- Apply temporal smoothing for video sequences

### Integration Points
- Designed to work with upstream face detection
- Outputs compatible with downstream mouth cropping
- Maintains coordinate system consistency

## Dependencies
- PyTorch
- OpenCV
- NumPy
- Pre-trained model weights (loaded automatically)

## Notes for Contributors
- Model weights are loaded automatically on first use
- GPU memory usage scales with batch size and input resolution
- The module supports both single image and batch processing
- JIT compilation provides speed improvements for repeated inference
- All models expect normalized input images (0-1 range)
