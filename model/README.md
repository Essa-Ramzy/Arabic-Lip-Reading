# model

## Purpose

The `model/` directory contains the core machine learning components for the Arabic lip reading system. This folder is responsible for:

- **Deep Learning Models**: Implementation of end-to-end video speech recognition (E2E-VSR) models
- **Neural Network Architectures**: Encoder modules including ResNet, TCN, DenseTCN, and Conformer implementations
- **Training and Inference**: Utility functions for data processing, augmentation, and model evaluation
- **ESPNet Integration**: Speech processing toolkit components for hybrid CTC/Attention architectures
- **Notebook Experiments**: Jupyter notebooks for model training, testing, and experimentation

## Folder Structure

```
model/
├── e2e_vsr.py                          # End-to-end video speech recognition model implementation
├── utils.py                            # Utility functions for data processing and training
├── master.ipynb                        # Main training and evaluation notebook
├── kaggle_master.ipynb                 # Kaggle environment training notebook
├── encoders/                           # Neural network encoder implementations
│   ├── encoder_models.py               # High-level encoder architectures
│   ├── pretrained_visual_frontend.pth  # Pretrained visual frontend weights
│   └── modules/                        # Core neural network building blocks
└── espnet/                             # ESPNet toolkit integration
    ├── encoder/                        # Conformer encoder implementations
    ├── decoder/                        # Transformer decoder components
    ├── transformer/                    # Core transformer building blocks
    ├── scorers/                        # Beam search scoring mechanisms
    └── *.py                            # Various ESPNet utilities and modules
```

## File Descriptions

### Core Model Components

**`e2e_vsr.py`**
- Main end-to-end video speech recognition model class
- Integrates visual frontend, temporal encoders, and sequence prediction
- Supports multiple encoder architectures (TCN, DenseTCN, Conformer)
- Handles both training and inference modes with CTC loss integration

**`utils.py`**
- `VideoDataset`: PyTorch dataset class for video and label loading
- `VideoAugmentation`: Kornia-based video augmentation pipeline with temporal masking
- `pad_packed_collate`: Custom collate function for variable-length sequences
- `compute_cer`/`compute_wer`: Character and Word Error Rate calculation functions
- `WarmupCosineScheduler`: Learning rate scheduler with warmup and cosine decay
- Text processing utilities for Arabic character handling and diacritics

### Training and Experimentation

**`master.ipynb`**
- Complete training pipeline notebook for local development
- Model configuration, data loading, training loops, and evaluation
- Visualization of training metrics and model performance
- Checkpoint saving and model export functionality

**`kaggle_master.ipynb`**
- Kaggle environment adaptation of the training pipeline
- Optimized for Kaggle's computational constraints and data access patterns
- Includes dataset mounting and environment setup specific to Kaggle

### Subdirectories

**`encoders/`**
- Contains custom encoder architectures and pretrained weights
- Implements VisualFrontend, DenseTCN, MultiscaleTCN, and VisualTemporalEncoder
- Modular design allows mixing and matching different encoder components

**`espnet/`**
- ESPNet toolkit integration for state-of-the-art speech processing
- Transformer-based encoders and decoders with attention mechanisms
- Beam search algorithms and scoring interfaces for inference
- CTC processing and prefix scoring for sequence alignment

## Internal Usage Tips

### Data Pipeline
- Use `VideoDataset` with appropriate augmentation for training
- Set `with_diaritics=True` for Arabic text with diacritical marks
- Apply normalization using global `MEAN` and `STD` values (0.421, 0.165)

### Model Training
- Initialize models with `e2e_vsr.E2EVSR` class
- Use `WarmupCosineScheduler` for optimal learning rate scheduling
- Apply `pad_packed_collate` for batching variable-length sequences

### Evaluation Metrics
- Use `compute_cer()` for character-level evaluation
- Use `compute_wer()` for word-level evaluation when applicable
- Both functions work directly with token indices from vocabulary

### Video Processing
- Input videos should be mouth-region cropped to 88x88 pixels
- Apply temporal augmentation during training for better generalization
- Handle variable sequence lengths through proper padding and masking

### Configuration
- Model architectures are configurable through encoder type selection
- Supports both diacritized and non-diacritized Arabic text processing
- Beam search parameters can be tuned for inference quality vs. speed trade-offs

### Dependencies
- **PyTorch**: Core deep learning framework
- **Kornia**: Computer vision library for video augmentation
- **ESPNet**: Speech processing toolkit
- **OpenCV**: Video file processing
- **editdistance**: Error rate calculation
- **pandas**: CSV label file processing