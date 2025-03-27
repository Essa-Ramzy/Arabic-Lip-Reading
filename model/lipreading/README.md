# Lipreading Module

This directory contains the core components of the Arabic lip reading system, including encoder models, preprocessing functions, transformer decoder, and utility functions.

## File Structure

```
lipreading/
├── models/                 # Backend neural network architectures
├── pretrained_frontend/    # Pretrained visual feature extractors
├── encoder_models.py       # Encoder implementations (TCN, DenseTCN, etc.)
├── optim_utils.py          # Optimization utilities for training
├── preprocess.py           # Video preprocessing functions
├── transformer_decoder.py  # Arabic-specific transformer decoder
└── utils.py                # Utility functions for training and evaluation
```

## Detailed File Explanations

### encoder_models.py

This file implements the encoder architectures for the lip reading system, responsible for processing visual features into temporal representations.

#### Key Components

1. **DenseTCN**: Dense Temporal Convolutional Network

   - Dense blocks with skip connections for feature reuse
   - Multi-scale temporal convolutions with different kernel sizes (3, 5, 7)
   - Configurable dilation rates for increasing receptive field
   - Optional squeeze-excitation blocks for channel-wise attention
   - Outputs sequence predictions for CTC loss

2. **MultiscaleTCN**: Multi-scale Temporal Convolutional Network

   - Processes features at multiple time scales simultaneously
   - Uses parallel convolutions with different kernel sizes
   - Efficiently aggregates multi-scale features
   - Configurable channel counts and activation functions

3. **Lipreading**: Main model class that integrates all components
   - Visual frontend using 3D convolutions for initial feature extraction
   - Backend (ResNet or ShuffleNet) for spatial feature processing
   - Temporal encoder (TCN, DenseTCN, or Conformer) for sequence modeling
   - Adapter layer to match dimensions between components
   - Supports different activation functions (default: Swish)
   - Option to extract features for downstream tasks

#### Implementation Details

- The `forward` method processes a batch through frontend, backbone, and encoder:

  ```python
  def forward(self, x, lengths):
      # Process through frontend, backbone, and encoder
      # Return sequence predictions
  ```

- The `_sequence_batch` function preserves sequence information for token-level prediction:

  ```python
  def _sequence_batch(x, lengths, B):
      # Keep sequences intact for CTC loss
      return x  # Shape (B, T, C)
  ```

- Integration with ESPNet's `ConformerEncoder` for transformer-based sequence modeling
- Weight initialization method for stable training:
  ```python
  def _initialize_weights_randomly(self):
      # Xavier initialization for different layer types
  ```

### optim_utils.py

This file provides optimization utilities for training the lip reading model, particularly focusing on learning rate scheduling.

#### Key Components

1. **CosineScheduler**: Implements cosine annealing learning rate schedule

   - Gradually reduces learning rate following a cosine curve
   - Starts from initial learning rate and decreases toward zero
   - Creates smooth learning rate transitions for better convergence

2. **get_optimizer**: Factory function that creates optimizers

   - Supports Adam, AdamW, and SGD optimizers
   - Configures appropriate weight decay for each optimizer type
   - Used in training scripts to instantiate optimizers

3. **change_lr_on_optimizer**: Helper function for learning rate updates
   - Directly modifies learning rate in all parameter groups
   - Used by CosineScheduler to implement the schedule

#### Usage

- The CosineScheduler is typically called at the end of each epoch:

  ```python
  scheduler = CosineScheduler(initial_lr, total_epochs)
  # During training loop:
  for epoch in range(total_epochs):
      train(...)
      scheduler.adjust_lr(optimizer, epoch)
  ```

- Differential learning rates can be implemented using parameter groups:
  ```python
  optimizer = optim.Adam([
      {'params': model.parameters(), 'lr': base_lr},
      {'params': decoder.parameters(), 'lr': base_lr * 1.5}
  ])
  ```

### preprocess.py

This file implements video preprocessing functions for data augmentation and normalization during training and inference.

#### Key Components

1. **Compose**: Container that applies multiple transforms sequentially

   - Similar to torchvision.transforms.Compose
   - Chains preprocessing steps together

2. **RgbToGray**: Converts RGB video frames to grayscale

   - Lip reading primarily relies on shape/motion, not color
   - Reduces model complexity and memory requirements

3. **Normalize**: Standardizes pixel values with mean and standard deviation

   - Improves training stability and convergence
   - Uses precomputed statistics from the dataset

4. **CenterCrop**: Crops frames to specified size from center

   - Focuses on the central region containing lips
   - Used primarily for evaluation/inference

5. **RandomCrop**: Random crop for data augmentation

   - Improves model generalization
   - Introduces position variations during training

6. **HorizontalFlip**: Randomly flips frames horizontally

   - Additional data augmentation technique
   - Controlled by flip_ratio parameter

7. **NormalizeUtterance**: Normalizes audio signals

   - Removes mean and divides by standard deviation
   - Applicable for audio-visual models

8. **AddNoise**: Adds controlled noise to audio signals

   - Configurable SNR levels for robustness
   - Used in audio-visual lip reading

9. **TimeMask**: Masks segments along temporal dimension
   - Similar to SpecAugment in audio processing
   - Improves temporal modeling robustness

#### Usage Examples

- **Training Augmentation Pipeline**:

  ```python
  train_transforms = Compose([
      RgbToGray(),
      RandomCrop(size=(88, 88)),
      HorizontalFlip(flip_ratio=0.5),
      Normalize(mean=0.4, std=0.2)
  ])
  ```

- **Evaluation Pipeline**:
  ```python
  eval_transforms = Compose([
      RgbToGray(),
      CenterCrop(size=(88, 88)),
      Normalize(mean=0.4, std=0.2)
  ])
  ```

### transformer_decoder.py

This file implements a custom transformer decoder for Arabic lip reading, specially designed for autoregressive sequence generation with Arabic text.

#### Key Components

1. **CustomDecoderLayer**: Modified decoder layer for Arabic generation

   - Enhanced cache handling for more robust autoregressive generation
   - Comprehensive error checking for mask and tensor dimensions
   - Supports pre-norm transformer architecture (normalize_before=True)
   - Implements both self-attention and cross-attention mechanisms

2. **ArabicTransformerDecoder**: Main decoder for Arabic lip reading
   - Token embedding layer with positional encoding
   - Memory projection to match encoder output dimension
   - Multiple transformer decoder layers
   - Output projection layer for vocabulary prediction
   - Special handling for SOS/EOS tokens in Arabic

#### Key Methods

1. **forward**: Main method for training

   ```python
   def forward(self, tgt, tgt_mask, memory, memory_mask=None):
       # Process target sequences through decoder
       # Return logits for next token prediction
   ```

2. **forward_one_step**: For single-step autoregressive generation

   ```python
   def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
       # Process single generation step with cache
       # Return logits and updated cache
   ```

3. **batch_beam_search**: Efficient parallel beam search
   ```python
   def batch_beam_search(self, memory, memory_mask=None, sos=None, eos=None,
                        blank=0, beam_size=5, penalty=0.0, maxlen=100,
                        minlen=0, ctc_weight=0.3):
       # Generate multiple hypotheses with beam search
       # Score with hybrid CTC/attention approach
       # Return best hypotheses for each batch item
   ```

#### Advanced Features

- **Memory Projection**: Dynamically adjusts to input feature dimension

  ```python
  def _create_memory_projection(self, input_dim):
      # Create projection layer that matches input dimension
  ```

- **Robust Error Handling**: Comprehensive validation of inputs

  ```python
  # Validation checks
  assert memory.size(0) == batch_size, f"Memory batch size {memory.size(0)} != input batch size {batch_size}"
  ```

- **Hybrid Scoring**: Combines CTC and attention scores
  ```python
  # Combine scores using CTC weight
  new_score = ctc_weight * new_ctc_score + (1.0 - ctc_weight) * new_att_score
  ```

### utils.py

This file contains utility functions used throughout the lipreading module, focusing on training, evaluation, and model management.

#### Key Components

1. **AverageMeter**: Tracks and computes running averages

   - Maintains current value, sum, count, and average
   - Used for monitoring metrics during training

2. **CheckpointSaver**: Manages model checkpoint saving

   - Saves latest checkpoint
   - Tracks and saves best-performing model
   - Optionally saves best model per learning rate step
   - Handles state tracking across training runs

3. **load_model**: Loads model and optimizer states

   - Supports optional size mismatch handling for transfer learning
   - Can load optimizer state for training resumption
   - Returns model, optimizer, epoch index, and full checkpoint

4. **IO Utilities**:

   - `read_txt_lines`: Reads lines from text file
   - `save_as_json`: Saves dictionary as formatted JSON
   - `load_json`: Loads and parses JSON file
   - `save2npz`: Saves compressed numpy array

5. **Logging Utilities**:

   - `get_logger`: Creates and configures logger with file and console output
   - `update_logger_batch`: Formats and logs batch progress
   - `get_save_folder`: Creates timestamped save directory

6. **Model Analysis**:
   - `calculateNorm2`: Calculates L2 norm of model parameters
   - `showLR`: Displays current learning rate

#### Usage Examples

- **Progress Tracking**:

  ```python
  losses = AverageMeter()
  batch_time = AverageMeter()
  # During training:
  losses.update(loss.item(), batch_size)
  ```

- **Checkpoint Management**:

  ```python
  saver = CheckpointSaver(save_dir, save_best_step=True, lr_steps=[30, 60])
  # After validation:
  saver.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
  }, current_acc, epoch)
  ```

- **Model Loading**:
  ```python
  model, optimizer, start_epoch, checkpoint = load_model(
      'checkpoints/best_model.pth',
      model,
      optimizer
  )
  ```

## Integration with Main System

This lipreading module serves as the core component of the Arabic lip reading system and integrates with:

1. The ESPNet speech processing toolkit for conformer and transformer components
2. The main training script (master_all.ipynb) which orchestrates the entire pipeline
3. The utility functions in the parent directory for data processing and evaluation
