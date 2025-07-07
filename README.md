# Arabic Lip Reading System - Backend API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)

A cutting-edge Arabic lip reading system powered by deep learning and computer vision. This repository contains the **backend API** built with FastAPI that handles video processing, lip reading inference, and AI-powered enhancements.

## 🎥 Demo

<div align="center">
  <video src="https://github.com/user-attachments/assets/b7620198-3be9-4938-86ec-2d20a9c57ddf" alt="Demo Video" style="border-radius: 15px; max-width: 80%;" controls />
</div>

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Demo](#demo)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Related Projects](#related-projects)
- [License](#license)

## 🚀 About the Project

This Arabic lip reading system leverages state-of-the-art deep learning models to convert lip movements in videos into Arabic text. The backend provides a robust API that:

- **Processes video uploads** with automatic face detection and lip region extraction
- **Performs lip reading inference** using advanced neural network models
- **Provides AI-powered enhancements** including text improvement, summarization, and translation
- **Handles real-time processing** with progress tracking and WebSocket support
- **Supports multiple Arabic dialects** and diacritization options

## 🛠️ Technologies Used

### Backend Framework

- **FastAPI** - High-performance web framework for building APIs
- **Python 3.8+** - Core programming language
- **PyTorch** - Deep learning framework for model inference
- **OpenCV** - Computer vision library for video processing

### AI & Machine Learning

- **RetinaFace** - Face detection and landmark extraction
- **Custom E2E VSR Model** - End-to-end video speech recognition
- **Gemini Pro** - AI-powered text enhancement and translation
- **ESPnet** - Speech processing toolkit

### Additional Tools

- **uvicorn** - ASGI server for running the API
- **Pydantic** - Data validation and serialization
- **python-multipart** - File upload handling
- **python-dotenv** - Environment variable management

## 📁 Project Structure

```
Arabic-Lib-Reading/
├── backend/                    # FastAPI backend application
│   ├── main.py                # Main application entry point
│   ├── video_processor.py     # Video processing and inference service
│   ├── gemini_service.py      # AI enhancement service
│   ├── localtunnel.py         # Local tunneling for external access
│   ├── uploads/               # Temporary video uploads
│   ├── processed/             # Processed video cache
│   ├── logs/                  # Application logs
│   └── .env                   # Environment configuration
├── model/                     # Deep learning models and utilities
│   ├── e2e_vsr.py            # End-to-end video speech recognition
│   ├── utils.py              # Utility functions and data processing
│   ├── encoders/             # Neural network encoders
│   └── espnet/               # ESPnet integration
├── preparation/              # Video preprocessing pipeline
│   └── retinaface/          # Face detection and cropping
└── README.md                # Project documentation
```

## 🔌 API Endpoints

### Core Endpoints

- **POST** `/transcribe` - Upload video for lip reading transcription
- **GET** `/transcribe/{task_id}` - Get transcription progress and results
- **POST** `/transcribe/preprocessed` - Process already preprocessed video
- **GET** `/health` - Health check endpoint

### Parameters

| Parameter     | Type    | Description                              |
| ------------- | ------- | ---------------------------------------- |
| `video`       | File    | Video file (MP4, AVI, MOV)               |
| `device`      | String  | Processing device (`cpu` or `cuda`)      |
| `model_name`  | String  | Encoder model (`resnet18`, `densenet3d`) |
| `diacritized` | Boolean | Enable Arabic diacritization             |
| `beam_size`   | Integer | Beam search size (1-10)                  |
| `enhance`     | Boolean | Enable AI text enhancement               |
| `summarize`   | Boolean | Generate content summary                 |
| `translate`   | String  | Target language for translation          |

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)
- Git

### Installation Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Essa-Ramzy/Arabic-Lip-Reading.git
   cd Arabic-Lib-Reading
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
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

```bash
# Build and run with Docker
docker build -t arabic-lip-reading .
docker run -p 8000:8000 arabic-lip-reading
```

## 🎯 Usage

### Basic Video Processing

```bash
curl -X POST "http://localhost:8000/transcribe/" \
  -F "video=@your_video.mp4" \
  -F "model_name=conformer" \
  -F "diacritized=true"
```

### Check Processing Status

```bash
curl -X GET "http://localhost:8000/progress/{task_id}/status"
```

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation powered by Swagger UI.

## 🔗 Related Projects

- **Frontend Application (Flutter)**: [https://github.com/Abdelrahman-Wael-1029/lip_reading.git](https://github.com/Abdelrahman-Wael-1029/lip_reading.git)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ESPnet team for the speech processing toolkit
- RetinaFace developers for face detection models
- Google Gemini team for AI enhancement capabilities
- The open-source community for various tools and libraries

---

<div align="center">
  <p>Built with ❤️ for Arabic language technology</p>
</div>
