# Dataset

## Purpose
The dataset directory contains the LRC-AR (Lip Reading Corpus - Arabic) dataset, which serves as the primary training and validation data for the Arabic Lip Reading API. This directory provides organized video-text pairs for training end-to-end visual speech recognition models on Arabic speech content.

## Folder Structure
```
dataset/
├── LRC-AR/                           # Main dataset directory
│   ├── Train/                        # Training data split
│   │   ├── Manually_Verified/        # High-quality manually verified data
│   │   │   ├── Video/                # MP4 video files (mouth region crops)
│   │   │   │   ├── 00139.mp4         # Video sample (5-digit ID format)
│   │   │   │   ├── 00154.mp4
│   │   │   │   └── ...               # ~1200+ manually verified videos
│   │   │   └── Csv/                  # Corresponding text labels
│   │   │       ├── 00139.csv         # Text transcript (word-level)
│   │   │       ├── 00154.csv
│   │   │       └── ...               # Matching CSV files for each video
│   │   └── Gemini_Transcribed/       # AI-transcribed data (larger volume)
│   │       ├── Video/                # MP4 video files
│   │       │   ├── [video_id].mp4    # Auto-transcribed video samples
│   │       │   └── ...               # ~10000+ AI-transcribed videos
│   │       └── Csv/                  # AI-generated text labels
│   │           ├── [video_id].csv    # Gemini-generated transcripts
│   │           └── ...               # Matching CSV files
│   └── Val/                          # Validation data split
│       └── Manually_Verified/        # High-quality validation data only
│           ├── Video/                # Validation video files
│           │   ├── [video_id].mp4    # Manually verified validation videos
│           │   └── ...               # ~300+ validation samples
│           └── Csv/                  # Validation text labels
│               ├── [video_id].csv    # Manual validation transcripts
│               └── ...               # Matching CSV files
└── README.md                         # This file
```

## File Descriptions

### Data Organization
- **Training Split**: Contains both manually verified (high-quality) and AI-transcribed (large-scale) data
- **Validation Split**: Contains only manually verified data for reliable evaluation
- **Video Files**: Preprocessed mouth region crops in MP4 format, ready for model input
- **CSV Files**: Word-level transcripts with Arabic text including diacritics

### Video Data Format
- **File Format**: MP4 video files
- **Content**: Cropped mouth regions from original speaker videos
- **Preprocessing**: Face detection → landmark extraction → mouth region cropping
- **Naming**: 5-digit numeric IDs (e.g., `00139.mp4`, `01234.mp4`)
- **Resolution**: Standardized mouth crop dimensions
- **Frame Rate**: Consistent temporal sampling

### Text Label Format
- **File Format**: CSV files with single column header `"word"`
- **Content**: One word per row, preserving Arabic diacritics
- **Language**: Modern Standard Arabic with full diacritization
- **Segmentation**: Word-level tokenization matching video temporal segments
- **Encoding**: UTF-8 to support Arabic script and diacritics

### Dataset Splits

#### Training Data
- **Manually_Verified**: ~1,200+ high-quality video-text pairs
  - Human-verified transcriptions
  - High annotation accuracy
  - Premium quality for model training
- **Gemini_Transcribed**: ~10,000+ AI-generated video-text pairs
  - Google Gemini AI transcriptions
  - Larger scale data for model robustness
  - Lower cost scaling approach

#### Validation Data
- **Manually_Verified**: ~300+ high-quality video-text pairs
  - Human-verified ground truth
  - Reliable evaluation benchmark
  - Consistent quality metrics

## Internal Usage

### Data Loading Pipeline
```python
# Example usage in training scripts
DATASET_ROOT = 'dataset'
DATASET_NAME = 'LRC-AR'

# Load training data (both manual and AI-transcribed)
train_manual_videos = f"{DATASET_ROOT}/{DATASET_NAME}/Train/Manually_Verified/Video/"
train_manual_labels = f"{DATASET_ROOT}/{DATASET_NAME}/Train/Manually_Verified/Csv/"
train_ai_videos = f"{DATASET_ROOT}/{DATASET_NAME}/Train/Gemini_Transcribed/Video/"
train_ai_labels = f"{DATASET_ROOT}/{DATASET_NAME}/Train/Gemini_Transcribed/Csv/"

# Load validation data (manual only)
val_videos = f"{DATASET_ROOT}/{DATASET_NAME}/Val/Manually_Verified/Video/"
val_labels = f"{DATASET_ROOT}/{DATASET_NAME}/Val/Manually_Verified/Csv/"
```

### Label Processing
```python
# Extract text from CSV files
def extract_label(csv_path, with_spaces=True, with_diaritics=True):
    # Read CSV and extract word-level tokens
    # Apply diacritic and spacing options
    # Return processed text sequence
    pass
```

### Video-Text Pairing
- Each video file has a corresponding CSV file with the same base filename
- Example: `00139.mp4` ↔ `00139.csv`
- Temporal alignment: Words in CSV correspond to video segments
- Quality consistency: Manual verification ensures accurate pairing

## Data Quality Characteristics

### Manually Verified Data
- **Accuracy**: Human-verified transcriptions with high precision
- **Consistency**: Standardized annotation guidelines
- **Coverage**: Diverse vocabulary and speaking styles
- **Quality**: Premium data for model fine-tuning

### AI-Transcribed Data
- **Scale**: Large volume for robust training
- **Coverage**: Broader vocabulary and domain coverage
- **Quality**: Good accuracy with some transcription errors
- **Efficiency**: Cost-effective data scaling approach

### Arabic Language Features
- **Diacritics**: Full diacritization (tashkeel) preserved
- **Script**: Arabic script with proper text encoding
- **Vocabulary**: Modern Standard Arabic words and phrases
- **Segmentation**: Word-level alignment with video timestamps

## Integration with Training Pipeline

### Curriculum Learning
- **Stage 1**: High-weight on manually verified data
- **Stage 2**: Balanced training on both data sources
- **Stage 3**: Full dataset utilization with appropriate weighting

### Data Augmentation
- Video augmentation applied during training
- Text normalization options (spaces, diacritics)
- Temporal alignment preservation

### Evaluation Protocol
- Validation performed exclusively on manually verified data
- Character Error Rate (CER) computation
- Word-level accuracy metrics

## Dependencies and Requirements
- Video processing: OpenCV, torchvision
- Text processing: UTF-8 encoding support
- Storage: ~50GB+ for complete dataset
- Arabic text rendering: Proper font support for visualization

## Notes for Contributors
- Dataset follows strict video ID → CSV pairing convention
- All Arabic text includes diacritics for precise pronunciation
- Validation split contains only manually verified data for reliable benchmarking
- Training supports both manual and AI-transcribed data with configurable weighting
- Video preprocessing ensures consistent mouth region crops across all samples