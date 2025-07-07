# model/encoders

## Purpose of the Folder

The `model/encoders/` directory contains the neural network encoder implementations for the Arabic lip reading system. This folder is responsible for:

- **Visual Feature Extraction**: Converting raw video frames into meaningful visual features
- **Temporal Modeling**: Capturing temporal dependencies in video sequences using advanced architectures
- **Multi-scale Processing**: Handling different temporal scales for robust lip reading recognition
- **Encoder Architectures**: Providing multiple encoder options (TCN, DenseTCN, Conformer) for different performance requirements

## Folder Structure

```
model/encoders/
├── encoder_models.py                # High-level encoder architecture definitions
├── pretrained_visual_frontend.pth   # Pretrained weights for visual frontend
└── modules/                         # Core neural network building blocks
    ├── densetcn.py                  # Dense Temporal Convolutional Networks
    ├── resnet.py                    # ResNet backbone implementations
    ├── se_module.py                 # Squeeze-and-Excitation attention module
    └── tcn.py                       # Temporal Convolutional Networks
```

## File Descriptions

### Core Architecture Files

**`encoder_models.py`**

- `VisualFrontend`: Initial visual feature extraction using 3D convolution + 2D ResNet backbone
- `DenseTCN`: Dense Temporal Convolutional Network encoder with dense connections
- `MultiscaleTCN`: Multi-branch TCN encoder for multi-scale temporal modeling
- `VisualTemporalEncoder`: High-level orchestrator combining visual frontend with temporal encoders

**`pretrained_visual_frontend.pth`**

- Pretrained weights for the VisualFrontend component
- Trained on large datasets for improved performance and faster convergence
- Can be loaded with `torch.load()` and used for transfer learning

### Neural Network Modules

**`modules/densetcn.py`**

- `DenseTemporalConvNet`: Dense temporal convolutional network implementation
- `_DenseBlock`: Dense connectivity blocks with feature concatenation
- `_ConvBatchChompRelu`: Multi-branch convolutional layers with different kernel sizes
- `TemporalConvLayer`: Basic temporal convolution building block
- `Chomp1d`: Causal padding removal for temporal convolutions

**`modules/resnet.py`**

- `ResNet`: ResNet architecture implementation (ResNet-18, ResNet-34)
- `BasicBlock`: Fundamental ResNet building block with residual connections
- `conv3x3`: 3x3 convolution helper function
- `downsample_basic_block`: Downsampling functions for residual connections

**`modules/se_module.py`**

- `SELayer`: Squeeze-and-Excitation attention mechanism
- Channel-wise feature recalibration for improved representation learning
- Adaptive pooling and fully connected layers for attention weights

**`modules/tcn.py`**

- `MultibranchTemporalConvNet`: Multi-branch TCN with parallel processing paths
- `TemporalConvNet`: Standard single-branch TCN implementation
- `MultibranchTemporalBlock`: Multi-scale temporal convolution block
- `TemporalBlock`: Basic temporal convolution block with residual connections
- `ConvBatchChompRelu`: Convolution-BatchNorm-Chomp-ReLU sequence

## Tips and Notes

### Model Selection

- **DenseTCN**: Best for capturing fine-grained temporal patterns with dense connections
- **MultiscaleTCN**: Optimal for multi-scale temporal modeling with parallel branches
- **Conformer**: Hybrid CNN-Transformer architecture for balanced performance

### Performance Considerations

- Use pretrained visual frontend weights for faster training convergence
- DenseTCN requires more memory due to dense connections but offers better gradient flow
- MultiscaleTCN provides better multi-scale modeling at the cost of computational complexity

### Integration

- All encoders are designed to work with ESPNet's CTC loss and beam search
- Output dimensions are configurable to match downstream decoder requirements
- Support for both diacritized and non-diacritized Arabic text recognition

### Configuration Options

- Configurable dropout rates for regularization
- Multiple activation functions (ReLU, PReLU, Swish/SiLU)
- Optional Squeeze-and-Excitation attention integration
- Flexible kernel sizes and dilation rates for temporal modeling

### Dependencies

- PyTorch for deep learning framework
- ESPNet for speech processing utilities
- NumPy for numerical operations
- Requires CUDA-compatible GPU for optimal performance

- `resnet.py`: Implements the ResNet architecture (e.g., ResNet-18, ResNet-34), including `BasicBlock` definitions, used as the backbone in the `VisualFrontend`.
- `tcn.py`: Provides the implementation for Multibranch Temporal Convolutional Networks (`MultibranchTemporalConvNet`), used by the `MultiscaleTCN` encoder.
- `densetcn.py`: Contains the implementation for Dense Temporal Convolutional Networks (`DenseTemporalConvNet`), serving as the core for the `DenseTCN` encoder.
- `se_module.py`: Implements the Squeeze-and-Excitation (SE) block, which can be optionally integrated into models like `DenseTCN` for channel-wise feature recalibration.

These modules provide the foundational layers for constructing the more complex encoder architectures.

## `pretrained_visual_frontend.pth`

This binary file contains pretrained weights specifically for the `VisualFrontend` component (which typically combines a 3D convolution layer with a ResNet). These weights are often learned from a large dataset and can significantly improve performance and reduce training time when used as a starting point for the VSR model.

To load these weights into your `VisualFrontend` instance:

```python
import torch

# Assuming `visual_frontend` is an instance of the VisualFrontend class
# and `device` is your target device (e.g., torch.device('cuda') or torch.device('cpu'))

state = torch.load('Working/model/encoders/pretrained_visual_frontend.pth', map_location=device)

# It's common for pretrained weights to be saved under a 'state_dict' key.
# Adjust the key if necessary based on how the weights were saved.
if 'state_dict' in state:
    visual_frontend.load_state_dict(state['state_dict'], strict=False)
elif 'model_state_dict' in state: # Another common key
    visual_frontend.load_state_dict(state['model_state_dict'], strict=False)
else:
    visual_frontend.load_state_dict(state, strict=False)

print("Pretrained visual frontend weights loaded.")
```

Using `strict=False` is often helpful to ignore mismatches if the model architecture has slight differences from when the weights were saved, or if only partial loading is intended (e.g., when fine-tuning).

These weights can be kept frozen during initial VSR model training or fine-tuned along with the rest of the model.
