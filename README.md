# Arabic Lip Reading System - Backend API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)

A cutting-edge Arabic lip reading system powered by deep learning and computer vision. This repository contains the **backend API** built with FastAPI that handles video processing, lip reading inference, and AI-powered enhancements.

## Table of Contents

- [About the Project](#about-the-project)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Related Projects](#related-projects)
- [License](#license)

## About the Project

This Arabic lip reading system leverages state-of-the-art deep learning models to convert lip movements in videos into Arabic text. The backend provides a robust API that:

- **Processes video uploads** with automatic face detection and lip region extraction
- **Performs lip reading inference** using advanced neural network models
- **Provides AI-powered enhancements** including text improvement, summarization, and translation
- **Handles real-time processing** with progress tracking via Server-Sent Events
- **Supports multiple Arabic dialects** and diacritization options

## Demo

<div align="center">
  <video src="https://github.com/user-attachments/assets/b7620198-3be9-4938-86ec-2d20a9c57ddf" alt="Demo Video" style="border-radius: 15px; max-width: 80%;" controls>
    <em>If you can't see the video, <a href="https://github.com/user-attachments/assets/b7620198-3be9-4938-86ec-2d20a9c57ddf">click here to watch it directly</a>.</em>
  </video>
</div>

## Technologies Used

### Backend Framework

- **FastAPI** - High-performance web framework for building APIs
- **Python 3.8+** - Core programming language
- **PyTorch** - Deep learning framework for model inference
- **OpenCV** - Computer vision library for video processing

### AI & Machine Learning

- **RetinaFace** - Face detection and landmark extraction
- **Custom E2E VSR Models** - End-to-end video speech recognition with multiple encoders:
  - **MSTCN** - Multiscale Temporal Convolutional Network
  - **DCTCN** - Dense Temporal Convolutional Network
  - **Conformer** - Conformer-based encoder architecture
- **Gemini Pro** - AI-powered text enhancement and translation
- **ESPnet** - Speech processing toolkit for transformer decoder

### Additional Tools

- **LocalTunnel** - Public tunnel for external access (using Node.js package)
- **uvicorn** - ASGI server for running the API
- **Pydantic** - Data validation and serialization
- **python-multipart** - File upload handling
- **python-dotenv** - Environment variable management
- **Google Generative AI** - AI enhancement and translation services

## Project Structure

```
Arabic-Lib-Reading/
├── backend/                                # FastAPI backend application
│   ├── main.py                             # Main application entry point
│   ├── video_processor.py                  # Video processing and inference service
│   ├── gemini_service.py                   # Google Gemini Pro AI enhancement service
│   ├── localtunnel.py                      # Local tunneling for external access
│   ├── kaggle_api_start.ipynb              # Kaggle environment startup notebook
│   ├── .env.example                        # Environment variables template
│   ├── package.json                        # Node.js dependencies for LocalTunnel
│   ├── uploads/                            # Temporary video uploads
│   ├── processed/                          # Processed video cache
│   └── logs/                               # Application logs
├── model/                                  # Deep learning models and utilities
│   ├── e2e_vsr.py                          # End-to-end video speech recognition
│   ├── utils.py                            # Utility functions and data processing
│   ├── master.ipynb                        # Main training and evaluation notebook
│   ├── kaggle_master.ipynb                 # Kaggle environment training notebook
│   ├── encoders/                           # Neural network encoders
│   │   ├── encoder_models.py               # High-level encoder architectures
│   │   ├── pretrained_visual_frontend.pth  # Pretrained weights
│   │   └── modules/                        # Core neural network building blocks
│   └── espnet/                             # ESPNet toolkit integration
│       ├── encoder/                        # Conformer encoder implementations
│       ├── decoder/                        # Transformer decoder components
│       ├── transformer/                    # Core transformer building blocks
│       ├── scorers/                        # Beam search scoring mechanisms
│       └── *.py                            # Various ESPNet utilities and modules
├── preparation/                            # Video preprocessing pipeline
│   └── retinaface/                         # Face detection and mouth region cropping
│       ├── detector.py                     # Main landmarks detector class
│       ├── mouth_cropping.py               # AVSR data loader and mouth cropping
│       ├── video_process.py                # Video preprocessing pipeline
│       ├── 20words_mean_face.npy           # Mean face template for alignment
│       └── ibug/                           # Face detection and alignment modules
│           ├── face_alignment/             # Facial landmark detection
│           └── face_detection/             # Face detection components
├── dataset/                                # Training and validation datasets
│   └── LRC-AR/                             # Arabic Lip Reading Corpus
│       ├── Train/                          # Training data split
│       │   ├── Manually_Verified/          # High-quality manual data
│       │   └── Gemini_Transcribed/         # AI-transcribed data
│       └── Val/                            # Validation data split
|           └── Manually_Verified/          # High-quality manual data
├── LICENSE                                 # MIT License
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

## API Endpoints

### Core Endpoints

- **POST** `/transcribe/` - Upload video for lip reading transcription
- **GET** `/progress/{task_id}` - Real-time progress tracking via Server-Sent Events
- **GET** `/progress/{task_id}/status` - Single status check for transcription task
- **DELETE** `/progress/{task_id}/cancel` - Cancel a running transcription task
- **POST** `/enhance-text/` - Enhance transcribed text with AI
- **GET** `/config` - Get API configuration and limits
- **GET** `/` - API information and available endpoints
- **GET** `/health` - Health check endpoint

### Parameters

| Parameter             | Type    | Description                                   |
| --------------------- | ------- | --------------------------------------------- |
| `file`                | File    | Video file (MP4, AVI, MOV, MKV, WebM)         |
| `file_hash`           | String  | Hash of previously processed video file       |
| `model_name`          | String  | Encoder model (`mstcn`, `dctcn`, `conformer`) |
| `landmark_model_name` | String  | Landmark model (`resnet50`, `mobilenet0.25`)  |
| `diacritized`         | Boolean | Enable Arabic diacritization                  |
| `beam_size`           | Integer | Beam search size (1-50, default: 10)          |
| `enhance`             | Boolean | Enable AI text enhancement                    |
| `include_summary`     | Boolean | Generate content summary                      |
| `include_translation` | Boolean | Include translation                           |
| `target_language`     | String  | Target language for translation               |

## Setup & Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- Git

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/username/Arabic-Lib-Reading.git
   cd Arabic-Lib-Reading
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Install Node.js dependencies for LocalTunnel
   cd backend
   npm install
   cd ..
   ```

4. **Set up environment variables:**

   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your configuration
   ```

5. **Download pre-trained models:**

   ```bash
   # Models will be downloaded automatically on first run
   # Or manually download from the releases page
   ```

6. **Run the application:**
   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

The API will be available at `http://localhost:8000`

### Docker Setup (Optional)

**Note:** Currently, no Dockerfile is provided. You can create one following the manual installation steps above.

## Usage

### Basic Video Processing

```bash
curl -X POST "http://localhost:8000/transcribe/" \
  -F "file=@your_video.mp4" \
  -F "model_name=conformer" \
  -F "diacritized=true"
```

### Check Processing Status

```bash
curl -X GET "http://localhost:8000/progress/{task_id}/status"
```

### Real-time Progress Tracking

```bash
# Server-Sent Events stream for real-time updates
curl -X GET "http://localhost:8000/progress/{task_id}"
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## Related Projects

- **Frontend Application (Flutter)**: [https://github.com/Abdelrahman-Wael-1029/lip_reading.git](https://github.com/Abdelrahman-Wael-1029/lip_reading.git)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ESPnet team for the speech processing toolkit
- RetinaFace developers for face detection models
- Google Gemini team for AI enhancement capabilities
- The open-source community for various tools and libraries

---

<div align="center">
  <p>Built with ❤️ for Arabic language technology</p>
</div>
