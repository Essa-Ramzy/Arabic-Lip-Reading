# Backend

## Purpose of the Folder

The `backend` folder contains the FastAPI-based web server that provides the core API functionality for the Arabic Lip Reading system. This folder is responsible for:

- **API Endpoints**: Handling video upload, processing, and transcription requests
- **Video Processing**: Orchestrating the lip reading pipeline using deep learning models
- **AI Enhancement**: Integrating with Google Gemini Pro for text improvement and translation
- **File Management**: Managing video uploads, processed files, and caching
- **Real-time Communication**: Providing progress updates and task management

## Folder Structure

```
backend/
├── main.py                     # FastAPI application entry point and API routes
├── video_processor.py          # Video preprocessing and lip reading inference
├── gemini_service.py           # Google Gemini Pro AI enhancement service
├── localtunnel.py              # Local tunneling setup for external access
├── .env                        # Environment variables (not in git)
├── .env.example                # Environment variables template
├── package.json                # Node.js dependencies for LocalTunnel
├── package-lock.json           # Node.js dependency lock file
├── uploads/                    # Temporary video uploads directory
├── processed/                  # Processed video cache directory
└── logs/                       # Application logs
    └── api.log                 # Main API log file
```

## File Descriptions

### Core Application Files

**`main.py`**

- FastAPI application instance and configuration
- API endpoint definitions for transcription, enhancement, and configuration
- Request/response models using Pydantic
- CORS middleware and error handling
- Server-Sent Events for real-time progress updates
- Task management and cancellation functionality

**`video_processor.py`**

- `VideoPreprocessor`: Handles video loading, orientation detection, and mouth region extraction
- `LipReadingPredictor`: Manages deep learning model inference for lip reading
- `VideoInferenceService`: Complete service combining preprocessing and prediction
- `ModelDownloader`: Downloads pre-trained models from Google Drive
- Integration with RetinaFace for face detection and VideoProcess for preprocessing

**`gemini_service.py`**

- `GeminiProService`: Google Gemini Pro API integration
- Text enhancement and correction functionality
- Content summarization and translation services
- Batch processing capabilities for multiple texts
- Error handling and response formatting

**`localtunnel.py`**

- LocalTunnel integration for exposing local server to the internet
- Automatic Node.js and LocalTunnel installation
- Cross-platform compatibility (Windows, Linux, macOS)
- Environment setup and dependency management
- Server startup orchestration

### Configuration Files

**`.env.example`**

- Template for environment variables
- Contains all required configuration keys with example values
- Includes model URLs, API keys, and server settings

**`package.json`**

- Node.js dependencies for LocalTunnel functionality
- Minimal configuration for tunnel setup

### Generated/Cache Directories

**`uploads/`**

- Temporary storage for uploaded video files
- Automatically cleaned up after processing
- Configurable maximum file size and allowed extensions

**`processed/`**

- Cache directory for preprocessed video files
- Enables hash-based reprocessing for improved performance
- Stores mouth region extracted videos

**`logs/`**

- Application logging directory
- Contains detailed API operation logs
- Configurable log levels and file rotation

## Tips and Notes

### Environment Setup

- Copy `.env.example` to `.env` and configure your settings
- Ensure `GOOGLE_AI_API_KEY` is set for AI enhancement features
- Model URLs in `.env` point to Google Drive files that are auto-downloaded

### Model Management

- Models are automatically downloaded on first use
- Supports multiple encoder types: MSTCN, DenseTCN, and Conformer
- Both diacritized and non-diacritized Arabic models available

### Performance Considerations

- Use CUDA device for faster inference if GPU is available
- Preprocessed video caching reduces repeat processing time
- LocalTunnel provides external access but may impact performance

### API Usage

- Interactive documentation available at `/docs` endpoint
- Real-time progress tracking via Server-Sent Events
- Task cancellation supported for long-running operations

### Development

- Use `uvicorn main:app --reload` for development with auto-reload
- Check `logs/api.log` for detailed operation information
- CORS is configured to allow all origins for development

### Dependencies

- Python 3.8+ required
- PyTorch for deep learning inference
- OpenCV for video processing
- FastAPI for web framework
- Node.js required for LocalTunnel functionality
