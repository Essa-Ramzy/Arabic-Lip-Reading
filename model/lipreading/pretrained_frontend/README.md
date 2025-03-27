# Pretrained Frontend

This directory contains the pretrained visual frontend components used for extracting features from video frames. The frontend is designed to process lip movements and extract meaningful features for the lip reading task.

## Directory Structure

```
pretrained_frontend/
├── encoder_models_pretrained.py  # Pretrained encoder model definitions
└── frontend.pth                  # Pretrained weights
```

## encoder_models_pretrained.py

This file contains the core model architecture definitions used for the pretrained visual frontend of the Arabic lip reading system. It includes the visual feature extraction pipeline and temporal encoding components.

### Overview

The file implements several key classes:

- A visual frontend for initial 3D feature extraction from video frames
- Multiple temporal encoding options (DenseTCN, MultiscaleTCN)
- The main Lipreading model that integrates all components

These components work together to process raw video frames of lip movements and produce feature sequences suitable for downstream sequence recognition.

### Key Components

#### 1. _sequence_batch Function

```python
def _sequence_batch(x, lengths, B):
    # Just return the sequence data properly shaped for CTC
    # Each item in batch will have sequence length based on its actual length
    return x  # Keep the sequence information intact - shape (B, T, C)
```

- Utility function used by temporal encoders
- Preserves sequence information for token-level prediction
- Maintains batch, time, and channel dimensions
- Essential for CTC-based training where sequence information must be preserved
- Unlike pooling functions, it doesn't collapse the temporal dimension

#### 2. DenseTCN Class

```python
class DenseTCN(nn.Module):
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                  kernel_size_set, dilation_size_set,
                  dropout, relu_type,
                  squeeze_excitation=False):
        # Implementation...
```

- Implements a dense temporal convolutional network architecture
- Key parameters:
  - `block_config`: Defines the number of layers in each dense block
  - `growth_rate_set`: Growth rate for each block (channel expansion)
  - `kernel_size_set` and `dilation_size_set`: Define the multi-scale properties
  - `squeeze_excitation`: Option to add SE blocks for attention
- Uses `DenseTemporalConvNet` from the models directory as the encoder trunk
- Handles dimension transformations for correct sequence processing
- Forward method:
  ```python
  def forward(self, x, lengths, B):
      x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T) for TCN
      out = self.encoder_trunk(x)  # Process through TCN
      out = out.transpose(1, 2)  # Back to (B, T, C)
      logits = self.encoder_output(out)  # Apply final linear layer
      return logits
  ```

#### 3. MultiscaleTCN Class

```python
class MultiscaleTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options,
                 dropout=0.2, relu_type='relu', dwpw=False):
        # Implementation...
```

- Implements a multi-scale temporal convolutional network
- Uses parallel convolutions with different kernel sizes (typically 3, 5, 7)
- Key parameters:
  - `input_size`: Dimension of input features
  - `num_channels`: List defining channel counts for each layer
  - `tcn_options`: Dictionary of options for customizing TCN behavior
  - `dwpw`: Option to use depthwise-pointwise convolutions for efficiency
- Uses `MultibranchTemporalConvNet` from models directory which processes features at multiple time scales
- Forward method handles dimension transformations similar to DenseTCN

#### 4. VisualFrontend Class

```python
class VisualFrontend(nn.Module):
    def __init__(self, frontend_nout=64, relu_type='prelu'):
        # Implementation...
```

- Core component for initial visual feature extraction
- Processes raw video frames using 3D convolutions
- Two main components:
  - `frontend3D`: Initial 3D convolution layers for spatio-temporal processing
  - `resnet_trunk`: ResNet backbone for deeper feature extraction
- Architecture details:
  - 3D convolution with kernel size (5,7,7) and stride (1,2,2)
  - Batch normalization and PReLU/ReLU activation
  - 3D max pooling with kernel size (1,3,3)
  - ResNet backbone with multiple blocks
- Forward method:
  ```python
  def forward(self, x):
      B, C, T, H, W = x.size()
      x = self.frontend3D(x)  # [B, frontend_nout, T, H//4, W//4]
      x = x.transpose(1, 2)   # [B, T, frontend_nout, H//4, W//4]
      x = x.contiguous().view(B * T, x.size(2), x.size(3), x.size(4))
      x = self.resnet_trunk(x)  # Process through ResNet
      x = x.view(B, T, -1)    # Reshape back to [B, T, features]
      return x
  ```
- Transforms 5D input (batch, channels, time, height, width) to 3D output (batch, time, features)

#### 5. Lipreading Class

```python
class Lipreading(nn.Module):
    def __init__(self,
                 modality='video',
                 hidden_dim=256,
                 backbone_type='resnet',
                 num_classes=500,
                 relu_type='swish',
                 tcn_options={},
                 densetcn_options={},
                 conformer_options={},
                 extract_feats=False):
        # Implementation...
```

- Main model class that integrates all components
- Highly configurable to support different temporal encoders:
  - DenseTCN
  - MultiscaleTCN
  - Conformer (from ESPNet)
- Key parameters:
  - `modality`: Input type ('video' for lip reading)
  - `hidden_dim`: Feature dimension for internal representations
  - `backbone_type`: Type of backbone network ('resnet' by default)
  - `relu_type`: Activation function type with 'swish' as default
  - Various options for different encoder types
- Architecture components:
  - Visual frontend for initial feature extraction
  - Optional adapter layer if dimensions don't match
  - Temporal encoder (TCN, DenseTCN, or Conformer)
  - Output layer for classification when using Conformer
- Forward method:

  ```python
  def forward(self, x, lengths):
      B, C, T, H, W = x.size()
      x = self.visual_frontend(x)  # Process through frontend
      x = self.adapter(x)          # Apply dimension adapter

      if self.extract_feats:
          return x  # Return features if requested

      # Process through temporal module
      if isinstance(self.encoder, ConformerEncoder):
          # Special handling for Conformer
          padding_mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(-2)
          x, _ = self.encoder(x, padding_mask)
          x = self.output_layer(x)
      else:
          # Forward through TCN or DenseTCN
          x = self.encoder(x, lengths, B)

      return x
  ```

- Also includes weight initialization method:
  ```python
  def _initialize_weights_randomly(self):
      # Xavier-like initialization for different layer types
  ```

### Implementation Details

#### Conformer Integration

- The Lipreading class integrates with ESPNet's ConformerEncoder:
  ```python
  self.encoder = ConformerEncoder(
      attention_dim=hidden_dim,
      attention_heads=conformer_options.get('attention_heads', 8),
      linear_units=conformer_options.get('linear_units', 2048),
      num_blocks=conformer_options.get('num_blocks', 6),
      dropout_rate=conformer_options.get('dropout_rate', 0.1),
      positional_dropout_rate=conformer_options.get('positional_dropout_rate', 0.1),
      attention_dropout_rate=conformer_options.get('attention_dropout_rate', 0.0),
      normalize_before=True,
      concat_after=False,
      macaron_style=True,
      use_cnn_module=True,
      cnn_module_kernel=conformer_options.get('cnn_module_kernel', 31),
  )
  ```
- Conformer combines self-attention with convolution for capturing both global and local dependencies
- Uses special padding mask handling to properly process variable-length sequences

#### Dimension Handling

- Careful handling of tensor dimensions throughout the pipeline:
  - Input: [B, C, T, H, W] (batch, channels, time, height, width)
  - After frontend: [B, T, features]
  - After temporal encoder: [B, T, num_classes]
- Transpositions between [B, C, T] and [B, T, C] for compatibility with different module expectations

#### Transfer Learning Support

- Extract features option allows using the model as a feature extractor:
  ```python
  if self.extract_feats:
      return x
  ```
- This facilitates transfer learning by extracting visual features for downstream tasks

## Usage in the System

The pretrained encoder models are used in the following ways:

1. **Visual Frontend Initialization**:

   ```python
   # Load pretrained frontend weights
   pretrained_path = 'lipreading/pretrained_frontend/frontend.pth'
   pretrained_weights = torch.load(pretrained_path, map_location=device)

   # Load weights into frontend
   model.visual_frontend.load_state_dict(pretrained_weights['state_dict'], strict=False)

   # Freeze frontend parameters
   for param in model.visual_frontend.parameters():
       param.requires_grad = False
   ```

2. **Feature Extraction**:

   ```python
   # Extract visual features
   with torch.no_grad():
       features = model(video_frames, lengths, extract_feats=True)
   ```

3. **Full Sequence Processing**:
   ```python
   # Process through entire model for prediction
   outputs = model(video_frames, lengths)
   ```

These pretrained models provide robust visual feature extraction, leveraging transfer learning from large-scale datasets to improve performance on the Arabic lip reading task.
