# model/encoders/modules

## Purpose of the Folder

The `model/encoders/modules/` directory contains the core neural network building blocks and fundamental layers used by the high-level encoder architectures. This folder is responsible for:

- **Basic Neural Network Components**: Implementing fundamental layers like temporal convolutions, ResNet blocks, and attention mechanisms
- **Temporal Modeling Primitives**: Providing building blocks for processing sequential video data
- **Attention Mechanisms**: Implementing Squeeze-and-Excitation modules for improved feature representation
- **Modular Architecture**: Offering reusable components that can be combined to create complex encoder architectures

## Folder Structure

```
model/encoders/modules/
├── README.md                    # This file, documenting the modules directory
├── densetcn.py                  # Dense Temporal Convolutional Network components
├── resnet.py                    # ResNet backbone implementations
├── se_module.py                 # Squeeze-and-Excitation attention module
└── tcn.py                       # Temporal Convolutional Network components
```

## File Descriptions

### Core Neural Network Components

**`densetcn.py`**
- `DenseTemporalConvNet`: Main Dense TCN implementation with dense connectivity
- `_DenseBlock`: Dense connectivity blocks that concatenate features from all previous layers
- `_ConvBatchChompRelu`: Multi-branch convolutional blocks with different kernel sizes
- `TemporalConvLayer`: Basic temporal convolution building block with batch normalization
- `_Transition`: Dimension reduction layers between dense blocks
- `Chomp1d`: Causal padding removal for maintaining temporal causality

**`resnet.py`**
- `ResNet`: Complete ResNet architecture (ResNet-18, ResNet-34) implementation
- `BasicBlock`: Fundamental ResNet building block with residual connections
- `conv3x3`: Helper function for 3x3 convolutions with padding
- `downsample_basic_block`: Standard downsampling for residual connections
- `downsample_basic_block_v2`: Enhanced downsampling using average pooling

**`se_module.py`**
- `SELayer`: Squeeze-and-Excitation attention mechanism implementation
- `_average_batch`: Utility function for batch averaging operations
- Channel-wise feature recalibration through adaptive pooling and fully connected layers

**`tcn.py`**
- `MultibranchTemporalConvNet`: Multi-branch TCN for parallel multi-scale processing
- `TemporalConvNet`: Standard single-branch TCN implementation
- `MultibranchTemporalBlock`: Multi-scale temporal convolution block with parallel branches
- `TemporalBlock`: Basic temporal convolution block with residual connections
- `ConvBatchChompRelu`: Convolution-BatchNorm-Chomp-ReLU sequence with optional depthwise separable convolutions

## Tips and Notes

### Architecture Design Patterns
- **Dense Connectivity**: DenseTCN modules use dense connections for better gradient flow and feature reuse
- **Residual Learning**: All temporal blocks include residual connections to enable deeper networks
- **Multi-scale Processing**: Parallel branches with different kernel sizes capture temporal patterns at various scales

### Performance Considerations
- **Memory Usage**: DenseTCN requires more memory due to feature concatenation but offers better gradient flow
- **Computational Complexity**: Multi-branch architectures provide better modeling at increased computational cost
- **Causal Convolutions**: All temporal convolutions maintain causality through proper padding and chomping

### Configuration Options
- **Activation Functions**: Support for ReLU, PReLU, and Swish/SiLU activations
- **Dropout Rates**: Configurable dropout for regularization in all modules
- **Kernel Sizes**: Flexible kernel size configurations for different temporal receptive fields
- **Dilation Rates**: Support for dilated convolutions to expand receptive fields efficiently

### Integration Guidelines
- **Modular Design**: All components are designed to be composable and reusable
- **Dimension Compatibility**: Modules handle dimension matching and projection automatically
- **Batch Processing**: All modules support batch processing for efficient training and inference

### Dependencies
- **PyTorch**: Core deep learning framework for all neural network components
- **torch.nn**: Standard PyTorch neural network modules
- **Collections**: OrderedDict for organized layer construction
- **Math**: Mathematical operations for weight initialization

### Usage Examples

#### Creating a Dense TCN Block
```python
from densetcn import DenseTemporalConvNet

# Create a DenseTCN with multiple dense blocks
model = DenseTemporalConvNet(
    block_config=[3, 3, 3],           # 3 dense blocks with 3 layers each
    growth_rate_set=[128, 256, 512],  # Growth rates for each block
    kernel_size_set=[3, 5, 7],        # Multi-scale kernels
    squeeze_excitation=True           # Enable SE attention
)
```

#### Setting up a ResNet Backbone
```python
from resnet import ResNet, BasicBlock

# Create ResNet-18 for visual feature extraction
backbone = ResNet(
    block=BasicBlock,
    layers=[2, 2, 2, 2],              # ResNet-18 configuration
    relu_type='prelu',                # Use PReLU activation
    dropout_rate=0.1                  # Add dropout for regularization
)
```

#### Using Multi-branch TCN
```python
from tcn import MultibranchTemporalConvNet

# Create multi-branch TCN for multi-scale modeling
tcn = MultibranchTemporalConvNet(
    num_inputs=512,
    num_channels=[256, 512, 768],     # Channel progression
    tcn_options={
        'kernel_size': [3, 5, 7, 9]   # Multi-scale kernels
    },
    dropout=0.2,
    relu_type='swish'                 # Use Swish activation
)
```
