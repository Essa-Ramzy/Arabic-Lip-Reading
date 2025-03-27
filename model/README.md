# Arabic Lip Reading Model Directory Structure

This directory contains the implementation of an Arabic lip reading system using a hybrid CTC/Attention architecture. The system consists of several key components that work together to perform lip reading on Arabic videos.

## Directory Structure

```
model/
├── espnet/                     # ESPNet components
├── lipreading/                 # Lip reading specific modules
├── master_all.ipynb            # Main training notebook
└── utils.py                    # Utility functions
```

## utils.py - Detailed Explanation

This file contains essential utility functions used throughout the Arabic lip reading system.

### File Structure

```python
import torch
import editdistance
from model.espnet.nets_utils import make_pad_mask
```

- **torch**: Main PyTorch library for tensor operations
- **editdistance**: Library for computing Levenshtein distance (edit distance)
- **make_pad_mask**: Imported from ESPNet for mask handling

### Function-by-Function Explanation

#### 1. `make_non_pad_mask`

```python
def make_non_pad_mask(lengths, xs=None, length_dim=-1):
    """Make mask tensor containing indices of non-padded part."""
    return ~make_pad_mask(lengths, xs, length_dim)
```

This function:

- Takes sequence lengths and optionally a reference tensor
- Creates a boolean mask with True values for actual sequence content and False for padding
- Inverts the result of `make_pad_mask` using the `~` operator
- Is critical for attention mechanisms to focus only on valid sequence parts

#### 2. `pad_packed_collate`

```python
def pad_packed_collate(batch):
    """Pads data and labels with different lengths in the same batch
    """
    data_list, input_lengths, labels_list, label_lengths = zip(*batch)
    c, max_len, h, w = max(data_list, key=lambda x: x.shape[1]).shape

    data = torch.zeros((len(data_list), c, max_len, h, w))

    # Only copy up to the actual sequence length
    for idx in range(len(data)):
        data[idx, :, :input_lengths[idx], :, :] = data_list[idx][:, :input_lengths[idx], :, :]

    # Flatten labels for CTC loss
    labels_flat = []
    for label_seq in labels_list:
        labels_flat.extend(label_seq)
    labels_flat = torch.LongTensor(labels_flat)

    # Convert lengths to tensor
    input_lengths = torch.LongTensor(input_lengths)
    label_lengths = torch.LongTensor(label_lengths)
    return data, input_lengths, labels_flat, label_lengths
```

This function:

- Unpacks the batch into separate components: video data, input lengths, labels, label lengths
- Determines the maximum sequence length and creates a zero-padded tensor
- Copies each video sequence up to its actual length
- Flattens all label sequences into a single tensor for CTC loss processing
- Converts all length data to LongTensors
- Returns a tuple containing the padded video data, input lengths, flattened labels, and label lengths
- Is used as the `collate_fn` parameter in PyTorch DataLoader

#### 3. `indices_to_text`

```python
def indices_to_text(indices, idx2char):
    """
    Converts a list of indices to text using the reverse vocabulary mapping.
    """
    try:
        return ''.join([idx2char.get(i, '') for i in indices])
    except UnicodeEncodeError:
        # Handle encoding issues in Windows console
        # Return a safe representation that won't cause encoding errors
        safe_text = []
        for i in indices:
            char = idx2char.get(i, '')
            try:
                # Test if character can be encoded
                char.encode('cp1252')
                safe_text.append(char)
            except UnicodeEncodeError:
                # Replace with a placeholder for characters that can't be displayed
                safe_text.append(f"[{i}]")
        return ''.join(safe_text)
```

This function:

- Takes predicted or reference indices and converts them to readable text
- Uses the `idx2char` mapping dictionary to convert each index to its character
- Handles UnicodeEncodeError exceptions that can occur with Arabic text in Windows environments
- Provides a fallback mechanism to replace non-displayable characters with their index values
- Is essential for displaying model predictions and computing text-based metrics

#### 4. `compute_cer`

```python
def compute_cer(reference_indices, hypothesis_indices):
    """
    Computes Character Error Rate (CER) directly using token indices.
    Takes raw token indices from our vocabulary (class_mapping.txt) rather than Unicode text.

    Returns a tuple of (CER, edit_distance)
    """
    # Use the indices directly - each index is one token in our vocabulary
    ref_tokens = reference_indices
    hyp_tokens = hypothesis_indices

    # Calculate edit distance using the editdistance library
    edit_distance = editdistance.eval(ref_tokens, hyp_tokens)

    # Calculate CER
    cer = edit_distance / max(len(ref_tokens), 1)  # Avoid division by zero

    return cer, edit_distance
```

This function:

- Takes reference (ground truth) and hypothesis (predicted) token indices
- Uses the `editdistance` library to compute the Levenshtein distance between sequences
- Computes the Character Error Rate (CER) by dividing the edit distance by the reference length
- Handles the edge case of empty reference sequences (avoids division by zero)
- Returns both the CER and raw edit distance as a tuple
- Is the primary metric for evaluating model performance in lip reading

### Usage Patterns

#### 1. DataLoader Setup

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    collate_fn=pad_packed_collate
)
```

#### 2. Mask Creation

```python
# Create memory mask based on actual encoder output lengths
memory_mask = torch.zeros((batch_size, encoder_features.size(1)), device=device).bool()
for b in range(batch_size):
    memory_mask[b, :output_lengths[b]] = True
```

#### 3. Text Conversion

```python
# Convert predictions to text
pred_text = indices_to_text(pred_indices, idx2char)
target_text = indices_to_text(target_idx, idx2char)
```

#### 4. Metric Calculation

```python
# Calculate and log CER
cer, edit_distance = compute_cer(target_idx, pred_indices)
print(f"CER: {cer:.4f}, Edit Distance: {edit_distance}")
```

## master_all.ipynb - Detailed Explanation

This Jupyter notebook is the main entry point for training and evaluating the Arabic lip reading system. It orchestrates the entire pipeline from data loading to model training and inference.

### 1. Imports and Initial Setup

```python
import torch, os, cv2, gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
import editdistance
```

- **Core libraries**: PyTorch for deep learning, OpenCV for video processing
- **Helper modules**: NumPy for numerical operations, Pandas for data handling
- **Utility imports**: GC for garbage collection, editdistance for CER calculation
- **Data processing**: DataLoader, transforms, train_test_split

```python
from lipreading.pretrained_frontend.encoder_models_pretrained import Lipreading
from lipreading.optim_utils import CosineScheduler
from lipreading.transformer_decoder import ArabicTransformerDecoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from utils import *
import logging
from datetime import datetime
```

- **Custom modules**: Lipreading model, scheduler, decoder
- **ESPNet components**: Masking utilities for transformers
- **Utility functions**: From utils.py (collation, masking, evaluation)
- **Logging setup**: Standard Python logging for tracking progress

```python
# Setup logging
log_filename = f'../Logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

- Creates unique timestamped log file for training session
- Sets up basic logging configuration with ISO format timestamps
- Logs at INFO level and higher for training progress tracking

### 2. Reproducibility Setup

```python
# Setting the seed for reproducibility
seed = 0
def reset_seed():
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

- **Seed setting**: Fixed seed (0) for reproducibility
- **reset_seed()**: Function to apply seed to all random number generators
- **CUDA settings**: Disables benchmarking and enables determinism for reproducibility
- **Device selection**: Automatically chooses GPU if available, otherwise falls back to CPU

### 3. Dataset Preparation

#### 3.1 Character Mapping

```python
def extract_label(file):
    label = []
    diacritics = {
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u06E2',  # Small High meem
    }

    sentence = pd.read_csv(file)
    for word in sentence.word:
        for char in word:
            if char not in diacritics:
                label.append(char)
            else:
                label[-1] += char

    return label
```

- Parses CSV files containing Arabic text
- Defines set of Arabic diacritical marks to handle properly
- Processes each character, keeping diacritics with their base character
- Returns list of characters with their diacritics

```python
classes = set()
for i in os.listdir('../Dataset/Csv (with Diacritics)'):
    file = '../Dataset/Csv (with Diacritics)/' + i
    label = extract_label(file)
    classes.update(label)

mapped_classes = {}
for i, c in enumerate(sorted(classes, reverse=True), 1):
    mapped_classes[c] = i
```

- Iterates through all CSV files to collect unique characters
- Creates a set of all unique characters in the dataset
- Maps each character to a unique integer index (starting from 1)
- Index 0 is reserved for blank/padding token in CTC

#### 3.2 Video Dataset Class

```python
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, label_paths, transform=None):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label_path = self.label_paths[index]
        frames = self.load_frames(video_path=video_path)
        label = torch.tensor(list(map(lambda x: mapped_classes[x], extract_label(label_path))))
        input_length = torch.tensor(frames.size(1), dtype=torch.long)
        label_length = torch.tensor(len(label), dtype=torch.long)
        return frames, input_length, label, label_length
```

- Standard PyTorch Dataset implementation for video data
- Initializes with paths to videos and corresponding labels
- **len** returns the number of video samples
- **getitem** loads frames and labels for a given index
- Converts text labels to tensor of indices using the mapping
- Returns tuple of (frames, input_length, label, label_length)

```python
def load_frames(self, video_path):
    frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_pil = Image.fromarray(frame, 'L')
            frames.append(frame_pil)

    if self.transform is not None:
        frames = [self.transform(frame) for frame in frames]
    frames = torch.stack(frames).permute(1, 0, 2, 3)
    return frames
```

- Helper method to load and preprocess video frames
- Opens video file using OpenCV's VideoCapture
- Extracts each frame and converts to grayscale
- Converts frames to PIL Image format for transforms
- Applies specified transforms if provided
- Stacks frames as tensor and rearranges dimensions to [C, T, H, W]

```python
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.419232189655303955078125, std=0.133925855159759521484375),
])
```

- Defines transformation pipeline for processing video frames
- Converts PIL images to PyTorch tensors
- Normalizes with pre-computed mean and standard deviation values

#### 3.3 Data Loading and Splitting

```python
videos_dir = "../Dataset/Preprocessed_Video"
labels_dir = "../Dataset/Csv (with Diacritics)"
videos, labels = [], []
file_names = [file_name[:-4] for file_name in os.listdir(videos_dir)]
for file_name in file_names:
    videos.append(os.path.join(videos_dir, file_name + ".mp4"))
    labels.append(os.path.join(labels_dir, file_name + ".csv"))
```

- Sets paths to video and label directories
- Creates lists to store paths to video files and label files
- Gets file names without extension from video directory
- Constructs full paths to corresponding video and label files

```python
X_temp, X_test, y_temp, y_test = train_test_split(videos, labels, test_size=0.1000, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=seed)
```

- Splits dataset into train/validation/test sets
- First split: 90% for combined train+validation, 10% for test
- Second split: Splits the 90% into 80% train, 10% validation
- Uses fixed random seed for reproducibility

```python
train_dataset = VideoDataset(X_train, y_train, transform=data_transforms)
val_dataset = VideoDataset(X_val, y_val, transform=data_transforms)
test_dataset = VideoDataset(X_test, y_test, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=pad_packed_collate)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn=pad_packed_collate)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn=pad_packed_collate)
```

- Creates dataset objects for train, validation, and test sets
- Creates data loaders with batch size 32
- Enables shuffling for training data only
- Uses pin_memory=True for faster GPU transfers
- Uses custom pad_packed_collate function to handle variable-length sequences

### 4. Model Configuration

#### 4.1 Vocabulary Setup

```python
# Build vocabulary setup
base_vocab_size = len(mapped_classes) + 1  # +1 for blank token (0)
sos_token_idx = base_vocab_size  # This places SOS after all normal tokens
eos_token_idx = base_vocab_size + 1  # This places EOS after SOS
full_vocab_size = base_vocab_size + 2  # +2 for SOS and EOS tokens

# Build reverse mapping for decoding
idx2char = {v: k for k, v in mapped_classes.items()}
idx2char[0] = ""  # Blank token for CTC
idx2char[sos_token_idx] = "<sos>"  # SOS token
idx2char[eos_token_idx] = "<eos>"  # EOS token
print(f"Total vocabulary size: {full_vocab_size}")
print(f"SOS token index: {sos_token_idx}")
print(f"EOS token index: {eos_token_idx}")
```

- Calculates vocabulary size based on unique characters plus special tokens
- Reserves index 0 for blank/padding token (used in CTC)
- Positions SOS and EOS tokens after all character tokens
- Creates reverse mapping from indices to characters
- Adds special tokens to reverse mapping
- Prints vocabulary statistics for reference

#### 4.2 Temporal Encoder Options

```python
# DenseTCN configuration (our default backbone)
densetcn_options = {
    'block_config': [3, 3, 3, 3],               # Number of layers in each dense block
    'growth_rate_set': [384, 384, 384, 384],    # Growth rate for each block (must be divisible by len(kernel_size_set))
    'reduced_size': 512,                        # Reduced size between blocks (must be divisible by len(kernel_size_set))
    'kernel_size_set': [3, 5, 7],               # Kernel sizes for multi-scale processing
    'dilation_size_set': [1, 2, 5],             # Dilation rates for increasing receptive field
    'squeeze_excitation': True,                 # Whether to use SE blocks for channel attention
    'dropout': 0.1                              # Dropout rate
}

# MSTCN configuration
mstcn_options = {
    'tcn_type': 'multiscale',
    'hidden_dim': 512,
    'num_channels': [171, 171, 171, 171],  # 4 layers with 171 channels each (divisible by 3)
    'kernel_size': [3, 5, 7],              # 3 kernels for multi-scale processing
    'dropout': 0.1,
    'stride': 1,
    'width_mult': 1.0,
}

# Conformer configuration
conformer_options = {
    'attention_dim': 512,            # Same as hidden_dim for consistency
    'attention_heads': 8,            # Number of attention heads
    'linear_units': 2048,            # Size of position-wise feed-forward
    'num_blocks': 6,                 # Number of conformer blocks
    'dropout_rate': 0.1,             # General dropout rate
    'positional_dropout_rate': 0.1,  # Dropout rate for positional encoding
    'attention_dropout_rate': 0.0,   # Dropout rate for attention
    'cnn_module_kernel': 31          # Kernel size for convolution module
}

# Choose temporal encoder type: 'densetcn', 'mstcn', or 'conformer'
TEMPORAL_ENCODER = 'conformer'
```

- Defines three options for temporal encoder architectures
- **DenseTCN**: Dense temporal convolutional network with skip connections
- **MSTCN**: Multi-scale temporal convolutional network
- **Conformer**: Transformer with convolution modules
- Each option has detailed hyperparameters with explanatory comments
- Final line selects which encoder type to use

#### 4.3 Model Initialization

```python
# Step 1: Initialize the model first
print(f"Initializing model with {TEMPORAL_ENCODER} temporal encoder...")
logging.info(f"Initializing model with {TEMPORAL_ENCODER} temporal encoder")

if TEMPORAL_ENCODER == 'densetcn':
    model = Lipreading(
        densetcn_options=densetcn_options,
        hidden_dim=512,
        num_classes=base_vocab_size,
        relu_type='swish'
    ).to(device)
elif TEMPORAL_ENCODER == 'mstcn':
    model = Lipreading(
        tcn_options=mstcn_options,
        hidden_dim=mstcn_options['hidden_dim'],
        num_classes=base_vocab_size,
        relu_type='swish'
    ).to(device)
elif TEMPORAL_ENCODER == 'conformer':
    model = Lipreading(
        conformer_options=conformer_options,
        hidden_dim=conformer_options['attention_dim'],
        num_classes=base_vocab_size,
        relu_type='swish'
    ).to(device)
else:
    raise ValueError(f"Unknown temporal encoder type: {TEMPORAL_ENCODER}")

print("Model initialized successfully.")
```

- Initializes the Lipreading model with selected temporal encoder
- Hidden dimension is set to 512 for all encoder types
- num_classes is set to base_vocab_size (for CTC output)
- Uses Swish activation function instead of ReLU
- Moves model to the appropriate device (GPU or CPU)
- Logs initialization progress and verifies success

```python
# Step 2: Load pretrained frontend weights
print("\nStep 4.2: Loading pretrained frontend weights...")
logging.info("Loading pretrained frontend weights")

pretrained_path = 'lipreading/pretrained_frontend/frontend.pth'
pretrained_weights = torch.load(pretrained_path, map_location=device)
print(f"Loaded pretrained weights from {pretrained_path}")

# Load weights into frontend
model.visual_frontend.load_state_dict(pretrained_weights['state_dict'], strict=False)
print("Successfully loaded pretrained weights")

# Freeze frontend parameters
for param in model.visual_frontend.parameters():
    param.requires_grad = False

print("Frontend frozen - parameters will not be updated during training")
logging.info("Successfully loaded and froze pretrained frontend")
```

- Loads pretrained weights for the visual frontend
- Maps weights to the correct device
- Applies weights to visual_frontend component of the model
- Freezes all parameters in the frontend
- Parameters with requires_grad=False won't be updated during training
- This implements transfer learning (use pretrained frontend, train rest of model)

#### 4.4 Decoder and Training Setup

```python
# Initialize transformer decoder
print("\nStep 4.3: Initializing transformer decoder and training components...")
transformer_decoder = ArabicTransformerDecoder(
    vocab_size=full_vocab_size,  # Use full vocab size that includes SOS/EOS
    attention_dim=512,          # Matching hidden_dim from the model
    attention_heads=8,          # 8 heads for better attention to different parts of sequence
    num_blocks=6,              # 6 transformer decoder layers
    dropout_rate=0.1
).to(device)
```

- Initializes the ArabicTransformerDecoder
- Uses full_vocab_size including special tokens (SOS/EOS)
- attention_dim matches the encoder output dimension
- 8 attention heads for parallel attention computation
- 6 transformer decoder blocks
- Dropout rate of 0.1 for regularization
- Moves decoder to the same device as the encoder

```python
# Training parameters
initial_lr = 3e-4
total_epochs = 80
scheduler = CosineScheduler(initial_lr, total_epochs)

# Loss functions
ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
ce_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad token

# Optimizer with different learning rates for encoder and decoder
optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': initial_lr},
    {'params': transformer_decoder.parameters(), 'lr': initial_lr * 1.5}  # Higher LR for transformer
])
```

- Sets initial learning rate to 3e-4
- Plans for 80 total training epochs
- Creates cosine learning rate scheduler for smooth decay
- Initializes CTC loss function with blank index 0
- Initializes cross-entropy loss that ignores padding tokens
- Creates Adam optimizer with different learning rates:
  - Base rate for encoder
  - 1.5x rate for decoder (needs to learn faster)

### 5. Training and Evaluation Functions

#### 5.1 RNG State Management

```python
def get_rng_state():
    state = {}
    try:
        state['torch'] = torch.get_rng_state()
        state['numpy'] = np.random.get_state()
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state()
        else:
            state['cuda'] = None

        # Validate RNG state types
        if not isinstance(state['torch'], torch.Tensor):
            print("Warning: torch RNG state is not a tensor, creating a valid state")
            state['torch'] = torch.random.get_rng_state()

    except Exception as e:
        print(f"Warning: Error capturing RNG state: {str(e)}. Using default state.")
        logging.warning(f"Error capturing RNG state: {str(e)}. Using default state.")
        # Create minimal valid state
        state = {
            'torch': torch.random.get_rng_state(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
    return state

def set_rng_state(state):
    try:
        if 'torch' in state and isinstance(state['torch'], torch.Tensor):
            torch.set_rng_state(state['torch'])
        if 'numpy' in state and state['numpy'] is not None:
            np.random.set_state(state['numpy'])
        if torch.cuda.is_available() and 'cuda' in state and state['cuda'] is not None:
            if isinstance(state['cuda'], torch.Tensor):
                torch.cuda.set_rng_state(state['cuda'])
    except Exception as e:
        print(f"Warning: Failed to set RNG state: {str(e)}")
        logging.warning(f"Failed to set RNG state: {str(e)}")
        print("Continuing with current RNG state")
        logging.info("Continuing with current RNG state")
```

- Functions to capture and restore random number generator states
- Ensures reproducibility across training runs
- Handles PyTorch, NumPy, and CUDA RNG states
- Includes comprehensive error handling
- Provides fallback mechanisms if state capture fails
- Used for checkpointing and resuming training

#### 5.2 Transformer Input Preparation

```python
def create_transformer_inputs(labels_flat, label_lengths, device):
    target_seqs = []
    start_idx = 0

    for b in range(label_lengths.size(0)):
        seq_len = label_lengths[b].item()
        seq = labels_flat[start_idx:start_idx + seq_len]
        target_seq = torch.cat([
            torch.tensor([sos_token_idx], device=device),
            seq,
            torch.tensor([eos_token_idx], device=device)
        ])
        target_seqs.append(target_seq)
        start_idx += seq_len

    # Pad sequences to same length
    max_len = max(len(seq) for seq in target_seqs)
    padded_seqs = []
    for seq in target_seqs:
        padded = torch.cat([seq, torch.zeros(max_len - len(seq), device=device, dtype=torch.long)])
        padded_seqs.append(padded)

    target_tensor = torch.stack(padded_seqs)

    # Teacher forcing with probability 0.5
    if torch.rand(1).item() < 0.5:
        # Teacher forcing: decoder input is target shifted right (remove last token)
        decoder_input = target_tensor[:, :-1]
    else:
        # No teacher forcing: decoder input is just the start token
        decoder_input = torch.full((target_tensor.size(0), 1), sos_token_idx, device=device)

    # Teacher forcing: decoder target is target shifted left (remove first token)
    decoder_target = target_tensor[:, 1:]

    # Create dynamic causal mask based on actual sequence length
    seq_len = decoder_input.size(1)
    batch_size = decoder_input.size(0)

    # Create causal mask that respects auto-regressive constraints
    tgt_mask = subsequent_mask(seq_len).to(device)  # Shape [seq_len, seq_len]

    # Ensure mask is 3D for attention modules: [batch_size, seq_len, seq_len]
    tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

    return decoder_input, decoder_target, tgt_mask
```

- Prepares inputs for transformer decoder from flattened label sequences
- Reconstructs individual sequences using label lengths
- Adds SOS and EOS tokens to each sequence
- Pads all sequences to the same length
- Implements teacher forcing with 50% probability:
  - With teacher forcing: input is target sequence with last token removed
  - Without teacher forcing: input is just the SOS token
- Sets decoder target as target sequence with first token removed
- Creates causal mask for autoregressive generation
- Returns decoder input, decoder target, and attention mask

#### 5.3 Training Loop

The training loop includes forward passes through the encoder and decoder, loss calculation, backpropagation, and memory management. Key components include:

- Setting models to training mode
- Processing each batch from train_loader
- Computing hybrid loss (CTC + cross-entropy)
- Backpropagating and updating parameters
- Memory management for efficient GPU utilization

#### 5.4 Evaluation Function

The evaluation function performs beam search decoding, computes Character Error Rate (CER), and reports metrics:

- Sets models to evaluation mode
- Disables gradient tracking
- Performs beam search decoding with CTC weight
- Converts predictions and targets to text
- Computes CER and other metrics

#### 5.5 Main Training Function

The main training function orchestrates the entire process:

- Handles checkpoint loading and saving
- Implements learning rate scheduling
- Calls train and evaluate functions
- Tracks best model based on validation performance
- Saves best model and periodic checkpoints

### 6. Training Execution

```python
reset_seed()
# Uncomment one of the following lines to run the full training or quick experiment
train_model(ctc_weight=0.2, checkpoint_path=None)
# quick_experiment(model, transformer_decoder, train_dataset, num_samples=50, ctc_weight=0.2)
```

- Resets random seed for reproducibility
- Calls train_model with CTC weight of 0.2
- Starts training from scratch (no checkpoint)
- Alternative quick_experiment function is commented out

## Training Flow

1. **Data Processing**:

   - Load video frames and Arabic text labels
   - Map Arabic characters to indices
   - Create train/validation/test splits
   - Apply data transformations

2. **Model Initialization**:

   - Load pretrained visual frontend
   - Initialize temporal encoder (Conformer/DenseTCN/MSTCN)
   - Setup Arabic-specific transformer decoder
   - Define loss functions and optimizer

3. **Training Loop**:

   - Process batches of video sequences
   - Extract visual features using encoder
   - Compute CTC loss on encoder outputs
   - Generate sequences with transformer decoder
   - Compute cross-entropy loss on decoder outputs
   - Update model parameters

4. **Evaluation**:

   - Perform beam search decoding
   - Calculate Character Error Rate (CER)
   - Monitor validation performance
   - Save checkpoints and best model

5. **Inference**:
   - Process input video
   - Extract visual features
   - Perform beam search with hybrid scoring
   - Convert indices to Arabic text

## Integration Points

- **Visual Frontend** → **Temporal Encoder** → **Transformer Decoder**
- **utils.py** provides supporting functions throughout the pipeline
- **ESPNet Components** provide transformer building blocks
- **Hybrid Loss** combines CTC and attention-based approaches
