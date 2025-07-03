from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import shutil
import os
import logging
import hashlib
import uuid
import asyncio
import json
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

from video_processor import VideoInferenceService
from gemini_service import GeminiProService

# Pydantic models for API documentation and response validation
class TranscriptionMetadata(BaseModel):
    """Metadata about the transcription process."""
    video_path: str = Field(..., description="Path to the input video file")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    model_name: str = Field(..., description="Encoder model used")
    diacritized: bool = Field(..., description="Whether diacritized text was requested")
    beam_size: int = Field(..., description="Beam search size used")
    success: bool = Field(..., description="Whether processing was successful")
    error: Optional[str] = Field(None, description="Error message if processing failed")
    video_shape: Optional[List[int]] = Field(None, description="Shape of processed video tensor")
    processed_video_path: Optional[str] = Field(None, description="Path to processed video file")
    processed_video_hash: Optional[str] = Field(None, description="Hash for the processed video file for future reference")
    was_preprocessed: bool = Field(False, description="Whether the input was already preprocessed (hash provided)")

class AIEnhancement(BaseModel):
    """AI enhancement result."""
    success: bool = Field(..., description="Whether enhancement was successful")
    content: Optional[str] = Field(None, description="Enhanced content")
    error: Optional[str] = Field(None, description="Error message if enhancement failed")

class AIEnhancements(BaseModel):
    """Collection of AI enhancements."""
    enhancement: Optional[AIEnhancement] = Field(None, description="Text enhancement result")
    summary: Optional[AIEnhancement] = Field(None, description="Content summary result")
    translation: Optional[Dict[str, Any]] = Field(None, description="Translation result with target language")
    error: Optional[str] = Field(None, description="General enhancement error")

class TranscriptionResponse(BaseModel):
    """Response model for video transcription."""
    raw_transcript: str = Field(..., description="Raw transcription from the lip reading model")
    enhanced_transcript: str = Field(..., description="AI-enhanced transcription (same as raw if no AI enhancement)")
    metadata: TranscriptionMetadata = Field(..., description="Processing metadata")
    enhancements: AIEnhancements = Field(..., description="AI enhancement results")

class TextEnhancementResponse(BaseModel):
    """Response model for text enhancement."""
    original_text: str = Field(..., description="Original input text")
    enhanced_text: str = Field(..., description="AI-enhanced text")
    enhancements: AIEnhancements = Field(..., description="AI enhancement results")

class UploadConfig(BaseModel):
    """Upload configuration."""
    max_size_mb: float = Field(..., description="Maximum file size in MB")
    max_size_bytes: int = Field(..., description="Maximum file size in bytes")
    allowed_extensions: List[str] = Field(..., description="Allowed file extensions")

class ProcessingConfig(BaseModel):
    """Processing configuration."""
    default_beam_size: int = Field(..., description="Default beam search size")
    max_beam_size: int = Field(..., description="Maximum allowed beam search size")

class ModelConfig(BaseModel):
    """Model configuration."""
    default_model: str = Field(..., description="Default encoder model")
    default_landmark_model: str = Field(..., description="Default landmark detection model")
    default_diacritized: bool = Field(..., description="Default diacritization setting")
    available_models: List[str] = Field(..., description="Available encoder models")
    available_landmark_models: List[str] = Field(..., description="Available landmark detection models")

class AIConfig(BaseModel):
    """AI service configuration."""
    gemini_available: bool = Field(..., description="Whether Gemini AI service is available")

class ConfigResponse(BaseModel):
    """API configuration response."""
    upload: UploadConfig = Field(..., description="File upload configuration")
    processing: ProcessingConfig = Field(..., description="Processing configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    ai: AIConfig = Field(..., description="AI service configuration")

class APIInfo(BaseModel):
    """API information."""
    message: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    features: List[str] = Field(..., description="List of API features")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints with descriptions")

class CancelTaskResponse(BaseModel):
    """Response model for task cancellation."""
    task_id: str = Field(..., description="ID of the cancelled task")
    message: str = Field(..., description="Cancellation status message")
    was_running: bool = Field(..., description="Whether the task was actively running when cancelled")
    success: bool = Field(..., description="Whether cancellation was successful")

# Load environment variables
load_dotenv()

# Get backend directory for absolute paths
BACKEND_DIR = Path(__file__).parent

# Configure logging with file output
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = BACKEND_DIR / Path(os.getenv("LOG_FILE", "logs/api.log"))

# Create logs directory if it doesn't exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Clear any existing handlers to avoid conflicts
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode='a', encoding='utf-8')
    ],
    force=True  # Force reconfiguration
)

# Create logger
logger = logging.getLogger(__name__)

# Set logging level for our logger specifically
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Ensure handlers flush immediately
for handler in logger.handlers:
    handler.flush()

# Test logging to ensure it's working
logger.info("=== Arabic Lip Reading API Starting ===")
logger.info(f"Log level: {LOG_LEVEL}")
logger.info(f"Log file: {LOG_FILE}")

# Force a flush to ensure the log messages are written
logging.getLogger().handlers[0].flush() if logging.getLogger().handlers else None

def flush_logs():
    """Force flush all log handlers to ensure messages are written to file."""
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()

app = FastAPI(
    title="Arabic Lip Reading API",
    description="""
    ## Arabic Lip Reading API
    
    A comprehensive Arabic lip reading system using deep learning models with AI-powered enhancements.
    
    ### Features
    - **Multi-encoder support**: Choose between MSTCN, DenseTCN, and Conformer models
    - **Landmark detection**: ResNet50 and MobileNet options for face landmark detection
    - **Diacritization**: Support for both diacritized and non-diacritized Arabic text
    - **AI Enhancement**: Google Gemini integration for text enhancement, summarization, and translation
    - **Real-time progress tracking**: Server-Sent Events for live progress updates
    - **Task cancellation**: Cancel running tasks gracefully with resource cleanup
    - **Automatic model downloading**: Models are downloaded from Google Drive on first use
    - **Per-request model initialization**: Better isolation and resource management
    - **File hash caching**: Reuse processed videos with hash for faster re-inference
    
    ### Supported Video Formats
    - MP4, AVI, MOV, MKV, WebM
    - Automatic orientation detection and correction
    - Frame rate and resolution adaptive processing
    
    ### API Endpoints
    - **POST /transcribe/**: Upload video or use hash for lip reading transcription
    - **GET /progress/{task_id}**: Stream real-time progress updates via Server-Sent Events
    - **GET /progress/{task_id}/status**: Get current progress status (one-time request)
    - **DELETE|POST /progress/{task_id}/cancel**: Cancel a running transcription task
    - **POST /enhance-text/**: Enhance Arabic text with AI (no video required)
    - **GET /config**: Get API configuration and available models
    - **GET /docs**: Interactive API documentation (Swagger UI)
    - **GET /redoc**: Alternative API documentation (ReDoc)
    
    ### Hash-based Processing
    - Upload a video once, get a hash for the processed version
    - Reuse the hash for different models, beam sizes, or AI enhancements
    - Significant performance improvement for re-processing
    """,
    version="1.0.0",
    contact={
        "name": "Arabic Lip Reading API",
        "url": "https://github.com/HazemAI/arabic-lip-reading",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    tags_metadata=[
        {
            "name": "transcription",
            "description": "Video lip reading and transcription operations",
        },
        {
            "name": "enhancement",
            "description": "AI-powered text enhancement operations",
        },
        {
            "name": "configuration",
            "description": "API configuration and information",
        },
    ]
)

# Log FastAPI startup
logger.info("FastAPI application initialized")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
gemini_service = None

def parse_file_size(size_str: str) -> int:
    """Parse file size string (e.g., '10MB', '5GB') to bytes."""
    if not size_str:
        return 10 * 1024 * 1024  # Default 10MB
    
    size_str = size_str.upper().strip()
    
    # Handle pure numbers (assume bytes)
    if size_str.isdigit():
        return int(size_str)
    
    # Parse size with units
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 * 1024,
        'GB': 1024 * 1024 * 1024
    }
    
    for unit, multiplier in units.items():
        if size_str.endswith(unit):
            try:
                size_value = float(size_str[:-len(unit)])
                return int(size_value * multiplier)
            except ValueError:
                break
    
    # Default fallback
    return 10 * 1024 * 1024  # Default 10MB

# Configuration
UPLOAD_DIR = BACKEND_DIR / Path(os.getenv("UPLOAD_DIR", "uploads"))
PROCESSED_DIR = BACKEND_DIR / Path(os.getenv("PROCESSED_DIR", "processed"))
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY", None)
MAX_UPLOAD_SIZE = parse_file_size(os.getenv("MAX_UPLOAD_SIZE", "10MB"))
ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS", ".mp4,.avi,.mov,.mkv,.webm").split(",")

# Model configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "conformer")  # Default to Conformer
DEFAULT_LANDMARK_MODEL = os.getenv("DEFAULT_LANDMARK_MODEL", "resnet50")  # Default to ResNet50
DEFAULT_DIACRITIZED = os.getenv("DEFAULT_DIACRITIZED", "true").lower() in ("true", "1", "yes")

# Beam search configuration
DEFAULT_BEAM_SIZE = int(os.getenv("DEFAULT_BEAM_SIZE", "10"))
MAX_BEAM_SIZE = int(os.getenv("MAX_BEAM_SIZE", "50"))

# Create directories
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Gemini service at startup if API key is available
if GOOGLE_AI_API_KEY:
    try:
        logger.info("Initializing Gemini Pro service")
        gemini_service = GeminiProService()
        logger.info("Gemini Pro service initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini service: {str(e)}")
        gemini_service = None
else:
    logger.info("No GOOGLE_AI_API_KEY provided, Gemini service will not be available")

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event."""
    logger.info("=== FASTAPI APPLICATION STARTUP ===")
    logger.info("Arabic Lip Reading API is starting up...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Processed directory: {PROCESSED_DIR}")
    logger.info(f"Max upload size: {MAX_UPLOAD_SIZE} bytes")
    logger.info(f"Default model: {DEFAULT_MODEL}")
    logger.info(f"Default landmark model: {DEFAULT_LANDMARK_MODEL}")
    logger.info(f"Gemini service available: {gemini_service is not None}")
    logger.info("=== STARTUP COMPLETE ===\n")
    flush_logs()  # Ensure startup logs are written

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event."""
    logger.info("=== FASTAPI APPLICATION SHUTDOWN ===")
    logger.info("Arabic Lip Reading API is shutting down...")
    logger.info("=== SHUTDOWN COMPLETE ===\n")

def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return the path."""
    logger.info(f"Starting file upload process for file: {file.filename}")
    
    if not file.filename:
        logger.error("No filename provided in upload")
        raise HTTPException(status_code=400, detail="No filename provided")
    
    original_name = file.filename
    logger.info(f"Original filename: {original_name}")

    # Use pathlib for file path operations
    file_path = Path(original_name)
    ext = file_path.suffix

    # Check file extension
    file_ext = ext.lower()
    logger.info(f"File extension: {file_ext}")
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.error(f"Unsupported file format: {file_ext}. Allowed: {ALLOWED_EXTENSIONS}")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end to get file size
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    logger.info(f"File size: {file_size} bytes ({file_size / (1024 * 1024):.2f} MB)")
    
    if file_size > MAX_UPLOAD_SIZE:
        max_size_mb = MAX_UPLOAD_SIZE / (1024 * 1024)
        file_size_mb = file_size / (1024 * 1024)
        logger.error(f"File too large: {file_size_mb:.1f}MB > {max_size_mb:.1f}MB")
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Size: {file_size_mb:.1f}MB, Max allowed: {max_size_mb:.1f}MB"
        )
    
    # Generate unique filename using UUID
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / unique_filename

    logger.info(f"Generated unique filename: {unique_filename}")
    logger.info(f"Saving file to: {save_path}")
    
    # Save the uploaded file
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved successfully: {save_path}")
        return str(save_path)
    except Exception as e:
        logger.error(f"Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def ensure_gemini_service():
    """Ensure Gemini service is initialized, initialize if needed."""
    global gemini_service
    
    if gemini_service is None and GOOGLE_AI_API_KEY:
        try:
            logger.info("Initializing Gemini Pro service")
            gemini_service = GeminiProService()
            logger.info("Gemini Pro service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini service: {str(e)}")
            # Don't raise exception - allow API to work without Gemini

def generate_file_hash(file_path: str) -> str:
    """Generate a hash for a processed file path."""
    # Use the relative path from processed directory to generate consistent hash
    processed_path = Path(file_path)
    if processed_path.is_absolute():
        # Get relative path from PROCESSED_DIR
        try:
            relative_path = processed_path.relative_to(PROCESSED_DIR)
        except ValueError:
            # If path is not under PROCESSED_DIR, use the filename
            relative_path = processed_path.name
    else:
        relative_path = processed_path
    
    # Generate hash from the relative path
    hash_input = str(relative_path).encode('utf-8')
    file_hash = hashlib.sha256(hash_input).hexdigest()[:16]  # Use first 16 chars for shorter hash
    return file_hash

def get_file_from_hash(file_hash: str) -> str:
    """Get the file path from a hash, checking if it exists in processed directory."""
    logger.info(f"Looking for file with hash: {file_hash}")
    
    # Search for files in processed directory
    for file_path in PROCESSED_DIR.iterdir():
        if file_path.is_file():
            current_hash = generate_file_hash(str(file_path))
            if current_hash == file_hash:
                logger.info(f"Found file for hash {file_hash}: {file_path}")
                return str(file_path)
    
    logger.error(f"No file found for hash: {file_hash}")
    raise HTTPException(status_code=404, detail=f"No processed file found for hash: {file_hash}")

def is_valid_hash(hash_string: str) -> bool:
    """Check if a string is a valid file hash (16 char hex string)."""
    if not hash_string or len(hash_string) != 16:
        return False
    try:
        int(hash_string, 16)  # Check if it's valid hex
        return True
    except ValueError:
        return False
    
async def _apply_ai_enhancements(raw_transcript: str, include_summary: bool, include_translation: bool, target_language: str) -> dict:
    """Apply AI enhancements to the transcription."""
    enhancements = {}
    
    try:
        # Enhance transcription
        enhancement_result = gemini_service.enhance_transcription(raw_transcript)
        enhancements["enhancement"] = {
            "success": enhancement_result.success,
            "content": enhancement_result.content,
            "error": enhancement_result.error
        }
        
        # Use enhanced text for further processing if available
        text_for_processing = enhancement_result.content if enhancement_result.success else raw_transcript
        
        # Add summary if requested
        if include_summary:
            summary_result = gemini_service.summarize_content(text_for_processing)
            enhancements["summary"] = {
                "success": summary_result.success,
                "content": summary_result.content,
                "error": summary_result.error
            }
        
        # Add translation if requested
        if include_translation:
            translation_result = gemini_service.translate_text(
                text_for_processing, 
                target_language=target_language
            )
            enhancements["translation"] = {
                "success": translation_result.success,
                "content": translation_result.content,
                "target_language": target_language,
                "error": translation_result.error
            }
            
    except Exception as e:
        logger.error(f"AI enhancement failed: {str(e)}")
        enhancements["error"] = str(e)
    
    return enhancements

# Progress tracking system
class ProgressTracker:
    """Thread-safe progress tracker for video processing."""
    
    def __init__(self):
        self.progress_data = {}
        self.cancelled_tasks = set()
        self.running_futures = {}  # task_id -> Future mapping
        self.lock = threading.Lock()
    
    def create_task(self, task_id: str, total_steps: int = 100) -> None:
        """Create a new progress tracking task."""
        with self.lock:
            self.progress_data[task_id] = {
                "progress": 0,
                "total_steps": total_steps,
                "current_step": "Starting...",
                "status": "processing",
                "start_time": time.time(),
                "last_update": time.time(),
                "error": None,
                "result": None,
                "cancelled": False
            }
    
    def update_progress(self, task_id: str, progress: int, step_description: str = None) -> None:
        """Update progress for a task."""
        with self.lock:
            if task_id in self.progress_data and not self.progress_data[task_id].get("cancelled", False):
                self.progress_data[task_id]["progress"] = min(progress, self.progress_data[task_id]["total_steps"])
                self.progress_data[task_id]["last_update"] = time.time()
                if step_description:
                    self.progress_data[task_id]["current_step"] = step_description
    
    def complete_task(self, task_id: str, result: Any = None) -> None:
        """Mark task as completed."""
        with self.lock:
            if task_id in self.progress_data:
                self.progress_data[task_id]["progress"] = self.progress_data[task_id]["total_steps"]
                self.progress_data[task_id]["status"] = "completed"
                self.progress_data[task_id]["current_step"] = "Completed"
                self.progress_data[task_id]["last_update"] = time.time()
                self.progress_data[task_id]["result"] = result
    
    def fail_task(self, task_id: str, error: str) -> None:
        """Mark task as failed."""
        with self.lock:
            if task_id in self.progress_data:
                self.progress_data[task_id]["status"] = "failed"
                self.progress_data[task_id]["current_step"] = "Failed"
                self.progress_data[task_id]["error"] = error
                self.progress_data[task_id]["last_update"] = time.time()
    
    def get_progress(self, task_id: str) -> Dict[str, Any]:
        """Get current progress for a task."""
        with self.lock:
            if task_id in self.progress_data:
                data = self.progress_data[task_id].copy()
                # Calculate elapsed time
                data["elapsed_time"] = time.time() - data["start_time"]
                # Calculate percentage
                data["percentage"] = (data["progress"] / data["total_steps"]) * 100
                return data
            return None
    
    def cleanup_task(self, task_id: str) -> None:
        """Remove task data to free memory."""
        with self.lock:
            if task_id in self.progress_data:
                del self.progress_data[task_id]
            if task_id in self.running_futures:
                del self.running_futures[task_id]
            self.cancelled_tasks.discard(task_id)
    
    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a running task."""
        with self.lock:
            if task_id not in self.progress_data:
                return {"success": False, "message": "Task not found", "was_running": False}
            
            task_data = self.progress_data[task_id]
            
            # Check if task is already completed or failed
            if task_data["status"] in ["completed", "failed"]:
                return {
                    "success": False, 
                    "message": f"Task already {task_data['status']}", 
                    "was_running": False
                }
            
            # Check if task is already cancelled
            if task_data.get("cancelled", False):
                return {
                    "success": False, 
                    "message": "Task already cancelled", 
                    "was_running": False
                }
            
            # Mark task as cancelled
            self.cancelled_tasks.add(task_id)
            self.progress_data[task_id]["cancelled"] = True
            self.progress_data[task_id]["status"] = "cancelled"
            self.progress_data[task_id]["current_step"] = "Cancelled by user"
            self.progress_data[task_id]["last_update"] = time.time()
            
            # Try to cancel the future if it exists
            if task_id in self.running_futures:
                future = self.running_futures[task_id]
                cancelled = future.cancel()
                if not cancelled:
                    # Future is already running, it will check the cancelled flag
                    pass
            
            return {
                "success": True, 
                "message": "Task cancelled successfully", 
                "was_running": True
            }
    
    def is_cancelled(self, task_id: str) -> bool:
        """Check if a task has been cancelled."""
        with self.lock:
            return task_id in self.cancelled_tasks or (
                task_id in self.progress_data and 
                self.progress_data[task_id].get("cancelled", False)
            )
    
    def register_future(self, task_id: str, future) -> None:
        """Register a future for a task to enable cancellation."""
        with self.lock:
            self.running_futures[task_id] = future

# Global progress tracker
progress_tracker = ProgressTracker()

async def process_video_background(
    task_id: str,
    video_file_path: str,
    model_name: str,
    landmark_model_name: str,
    diacritized: bool,
    beam_size: int,
    is_preprocessed: bool,
    file_hash: Optional[str] = None,
    enhance: bool = False,
    include_summary: bool = False,
    include_translation: bool = False,
    target_language: str = "English"
):
    """
    Background processing function for video transcription with progress tracking.
    
    This function runs the video processing pipeline in the background and updates
    progress at each step, compatible with Flutter EventSource for real-time updates.
    """
    try:
        logger.info(f"Starting background processing for task: {task_id}")
        
        # Step 1: Initialize video service (10%)
        progress_tracker.update_progress(task_id, 5, "Initializing video service...")
        await asyncio.sleep(0.1)  # Allow other tasks to run
        
        # Check for cancellation
        if progress_tracker.is_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled during initialization")
            return
        
        try:
            logger.info(f"Initializing video service with model: {model_name}, landmark_model: {landmark_model_name}, diacritized: {diacritized}")
            service = VideoInferenceService(
                model_name=model_name,
                landmark_model_name=landmark_model_name,
                dia=diacritized
            )
            logger.info("Video inference service initialized successfully")
            progress_tracker.update_progress(task_id, 15, "Video service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize video service: {str(e)}")
            progress_tracker.fail_task(task_id, f"Video service initialization failed: {str(e)}")
            return

        # Step 2: Video preprocessing check (15%)
        if progress_tracker.is_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled before preprocessing")
            return
            
        if is_preprocessed:
            progress_tracker.update_progress(task_id, 25, "Using preprocessed video (skipping preprocessing)...")
        else:
            progress_tracker.update_progress(task_id, 20, "Video preprocessing will be performed...")
        
        await asyncio.sleep(0.1)
        
        # Step 3: Start video processing (20-70%)
        if progress_tracker.is_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled before video processing")
            return
            
        progress_tracker.update_progress(task_id, 30, "Starting video processing...")
        
        # Run the actual video processing in a thread to avoid blocking
        def run_video_processing():
            # Check for cancellation within the processing function
            if progress_tracker.is_cancelled(task_id):
                raise RuntimeError("Task was cancelled")
            
            # Define cancellation callback for video processor
            def cancellation_callback():
                return progress_tracker.is_cancelled(task_id)
            
            return service.process_video(
                video_file_path, 
                beam_size=beam_size,
                is_preprocessed=is_preprocessed,
                cancellation_callback=cancellation_callback
            )
        
        # Update progress during processing (simulate intermediate steps)
        progress_updates = [
            (40, "Analyzing video frames..."),
            (50, "Detecting face landmarks..."),
            (60, "Extracting mouth regions..."),
            (70, "Running lip reading inference...")
        ]
        
        # Start video processing in executor to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_video_processing)
            progress_tracker.register_future(task_id, future)
            
            # Update progress while processing
            update_idx = 0
            while not future.done():
                # Check for cancellation
                if progress_tracker.is_cancelled(task_id):
                    logger.info(f"Task {task_id} was cancelled during processing")
                    future.cancel()  # Try to cancel the future
                    return
                    
                if update_idx < len(progress_updates):
                    progress, message = progress_updates[update_idx]
                    progress_tracker.update_progress(task_id, progress, message)
                    update_idx += 1
                await asyncio.sleep(2)  # Update every 2 seconds
            
            # Get the result
            try:
                raw_transcript, metadata = future.result()
            except RuntimeError as e:
                if "cancelled" in str(e).lower():
                    logger.info(f"Task {task_id} processing was cancelled")
                    return
                raise
        
        progress_tracker.update_progress(task_id, 75, "Video processing completed")
        
        # Check for cancellation after video processing
        if progress_tracker.is_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled after video processing")
            return
        
        # Step 4: Process results (75-80%)
        video_hash = None
        if not is_preprocessed:
            # Generate hash for newly processed file
            try:
                processed_hash = generate_file_hash(PROCESSED_DIR / Path(video_file_path).name)
                video_hash = processed_hash
                logger.info(f"Generated hash for processed file: {processed_hash}")
            except Exception as e:
                logger.warning(f"Failed to generate hash: {e}")
                video_hash = file_hash
        else:
            # Use the provided hash
            video_hash = file_hash
        
        logger.info(f"Inference completed. Success: {metadata['success']}")
        if metadata['success']:
            logger.info(f"Raw transcript: '{raw_transcript}' (length: {len(raw_transcript)})")
        else:
            logger.error(f"Inference failed: {metadata.get('error', 'Unknown error')}")
        
        if not metadata['success']:
            progress_tracker.fail_task(task_id, f"Video processing failed: {metadata.get('error', 'Unknown error')}")
            return
        
        progress_tracker.update_progress(task_id, 80, "Preparing transcription results...")
        
        # Step 5: Prepare base response (80-85%)
        response = {
            "raw_transcript": raw_transcript,
            "enhanced_transcript": raw_transcript,
            "video_hash": video_hash,
            "metadata": metadata,
            "enhancements": {}
        }
        
        progress_tracker.update_progress(task_id, 85, "Base transcription completed")
        
        # Check for cancellation before AI enhancements
        if progress_tracker.is_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled before AI enhancements")
            return
        
        # Step 6: Apply AI enhancements if requested (85-95%)
        if enhance and gemini_service and raw_transcript:
            logger.info("Applying AI enhancements...")
            progress_tracker.update_progress(task_id, 90, "Applying AI enhancements...")
            
            # Check for cancellation before AI processing
            if progress_tracker.is_cancelled(task_id):
                logger.info(f"Task {task_id} was cancelled before AI enhancements")
                return
            
            try:
                enhancements = await _apply_ai_enhancements(
                    raw_transcript, include_summary, include_translation, target_language
                )
                response["enhancements"] = enhancements
                
                # Update enhanced transcript if enhancement was successful
                if enhancements.get("enhancement", {}).get("success"):
                    response["enhanced_transcript"] = enhancements["enhancement"]["content"]
                    logger.info("AI enhancement completed successfully")
                    progress_tracker.update_progress(task_id, 95, "AI enhancement completed")
                else:
                    logger.warning("AI enhancement failed or returned no results")
                    progress_tracker.update_progress(task_id, 95, "AI enhancement completed (with warnings)")
                    
            except Exception as e:
                logger.error(f"AI enhancement failed: {e}")
                response["enhancements"] = {"error": f"AI enhancement failed: {str(e)}"}
                progress_tracker.update_progress(task_id, 95, "AI enhancement failed, continuing...")
                
        elif enhance and not gemini_service:
            logger.warning("AI enhancement requested but Gemini service not available")
            response["enhancements"] = {"error": "Gemini service not available"}
            progress_tracker.update_progress(task_id, 95, "AI enhancement unavailable")
        else:
            progress_tracker.update_progress(task_id, 95, "Skipping AI enhancement")
        
        # Final cancellation check before completion
        if progress_tracker.is_cancelled(task_id):
            logger.info(f"Task {task_id} was cancelled before completion")
            return
        
        # Step 7: Complete task (100%)
        progress_tracker.update_progress(task_id, 100, "Processing completed successfully")
        progress_tracker.complete_task(task_id, response)
        
        logger.info(f"=== BACKGROUND PROCESSING COMPLETED SUCCESSFULLY FOR TASK: {task_id} ===")
        
    except Exception as e:
        logger.error(f"Unexpected error in background processing for task {task_id}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        progress_tracker.fail_task(task_id, f"Unexpected error: {str(e)}")
    finally:
        logger.info(f"=== BACKGROUND PROCESSING ENDED FOR TASK: {task_id} ===\n")
        flush_logs()  # Ensure all logs are written

@app.get("/progress/{task_id}")
async def get_progress_stream(task_id: str):
    """
    Stream progress updates for a task using Server-Sent Events.
    
    This endpoint is compatible with Flutter's EventSource for real-time progress updates.
    """
    async def event_generator():
        try:
            while True:
                progress_data = progress_tracker.get_progress(task_id)
                
                if progress_data is None:
                    # Task not found
                    yield f"data: {json.dumps({'error': 'Task not found', 'task_id': task_id})}\n\n"
                    break
                
                # Send progress update
                yield f"data: {json.dumps(progress_data)}\n\n"
                
                # If task is completed or failed, send final update and close
                if progress_data["status"] in ["completed", "failed"]:
                    await asyncio.sleep(1)  # Give client time to receive final update
                    break
                
                # Wait 1 second before next update
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in progress stream for task {task_id}: {e}")
            yield f"data: {json.dumps({'error': str(e), 'task_id': task_id})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/progress/{task_id}/status")
async def get_progress_status(task_id: str):
    """
    Get current progress status for a task (one-time request, not streaming).
    """
    progress_data = progress_tracker.get_progress(task_id)
    
    if progress_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return progress_data

@app.delete(
    "/progress/{task_id}/cancel",
    response_model=CancelTaskResponse,
    tags=["transcription"],
    summary="Cancel a running transcription task (DELETE method)",
    description="""
    Cancel a currently running or pending transcription task using DELETE method.
    
    **Note:** This endpoint is also available as POST /progress/{task_id}/cancel
    for better client compatibility.
    
    **Features:**
    - Immediate cancellation of pending tasks
    - Graceful cancellation of running tasks at next checkpoint
    - Safe cleanup of resources and temporary files
    - Clear status reporting
    
    **Use Cases:**
    - User wants to stop a long-running transcription
    - Error recovery when a task gets stuck
    - Resource management when starting new tasks
    - UI responsiveness for cancel operations
    
    **Cancellation Behavior:**
    - **Pending tasks**: Cancelled immediately before processing starts
    - **Running tasks**: Cancelled at the next safety checkpoint during processing
    - **AI enhancement**: Can be cancelled before or during AI processing
    - **Completed/Failed tasks**: Cannot be cancelled (returns appropriate message)
    
    **Flutter Example:**
    ```dart
    // Cancel a task using DELETE (preferred)
    final response = await http.delete(
      Uri.parse('$baseUrl/progress/$taskId/cancel'),
    );
    
    // Alternative: Cancel using POST (for compatibility)
    final response = await http.post(
      Uri.parse('$baseUrl/progress/$taskId/cancel'),
    );
    
    if (response.statusCode == 200) {
      final result = json.decode(response.body);
      if (result['success']) {
        print('Task cancelled: ${result['message']}');
      } else {
        print('Cannot cancel: ${result['message']}');
      }
    }
    ```
    
    **Python Example:**
    ```python
    # Using DELETE (preferred)
    response = requests.delete(f'/progress/{task_id}/cancel')
    
    # Alternative: Using POST (for compatibility)
    response = requests.post(f'/progress/{task_id}/cancel')
    
    result = response.json()
    if result['success']:
        print(f"Task {task_id} cancelled successfully")
    else:
        print(f"Cannot cancel task {task_id}: {result['message']}")
    ```
    """,
    responses={
        200: {"description": "Cancellation request processed (check success field for actual result)"},
        404: {"description": "Task not found"}
    }
)
async def cancel_task(task_id: str):
    """
    Cancel a running transcription task.
    """
    # Check if task exists
    progress_data = progress_tracker.get_progress(task_id)
    if progress_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Attempt to cancel the task
    cancel_result = progress_tracker.cancel_task(task_id)
    
    logger.info(f"Cancellation request for task {task_id}: {cancel_result}")
    
    return CancelTaskResponse(
        task_id=task_id,
        message=cancel_result["message"],
        was_running=cancel_result["was_running"],
        success=cancel_result["success"]
    )

@app.post(
    "/progress/{task_id}/cancel",
    response_model=CancelTaskResponse,
    tags=["transcription"],
    summary="Cancel a running transcription task (POST method)",
    description="""
    Cancel a currently running or pending transcription task using POST method.
    
    This is an alternative to the DELETE method for better client compatibility,
    especially with frameworks that have limited HTTP method support.
    
    **Features:**
    - Immediate cancellation of pending tasks
    - Graceful cancellation of running tasks at next checkpoint
    - Safe cleanup of resources and temporary files
    - Clear status reporting
    
    **Use Cases:**
    - User wants to stop a long-running transcription
    - Error recovery when a task gets stuck
    - Resource management when starting new tasks
    - UI responsiveness for cancel operations
    
    **Cancellation Behavior:**
    - **Pending tasks**: Cancelled immediately before processing starts
    - **Running tasks**: Cancelled at the next safety checkpoint during processing
    - **AI enhancement**: Can be cancelled before or during AI processing
    - **Completed/Failed tasks**: Cannot be cancelled (returns appropriate message)
    
    **Flutter Example:**
    ```dart
    // Cancel a task using POST
    final response = await http.post(
      Uri.parse('$baseUrl/progress/$taskId/cancel'),
    );
    
    if (response.statusCode == 200) {
      final result = json.decode(response.body);
      if (result['success']) {
        print('Task cancelled: ${result['message']}');
      } else {
        print('Cannot cancel: ${result['message']}');
      }
    }
    ```
    
    **Python Example:**
    ```python
    response = requests.post(f'/progress/{task_id}/cancel')
    result = response.json()
    
    if result['success']:
        print(f"Task {task_id} cancelled successfully")
    else:
        print(f"Cannot cancel task {task_id}: {result['message']}")
    ```
    """,
    responses={
        200: {"description": "Cancellation request processed (check success field for actual result)"},
        404: {"description": "Task not found"}
    }
)
async def cancel_task_post(task_id: str):
    """
    Cancel a running transcription task (POST method version).
    
    This endpoint provides the same functionality as DELETE /progress/{task_id}/cancel
    but uses POST method for better client compatibility.
    """
    # Check if task exists
    progress_data = progress_tracker.get_progress(task_id)
    if progress_data is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Attempt to cancel the task
    cancel_result = progress_tracker.cancel_task(task_id)
    
    logger.info(f"Cancellation request (POST) for task {task_id}: {cancel_result}")
    
    return CancelTaskResponse(
        task_id=task_id,
        message=cancel_result["message"],
        was_running=cancel_result["was_running"],
        success=cancel_result["success"]
    )

@app.post(
    "/transcribe/", 
    tags=["transcription"],
    summary="Start video transcription with progress tracking",
    description="""
    Start Arabic lip reading transcription process and return a task ID for progress tracking.
    
    This endpoint immediately returns a task ID that can be used to track progress via:
    - GET /progress/{task_id} - Server-Sent Events stream (recommended for Flutter)
    - GET /progress/{task_id}/status - Single status check
    
    **Process:**
    1. Submit transcription request (file or hash)
    2. Receive task_id in response immediately
    3. Connect to /progress/{task_id} for real-time updates
    4. Process completes with final result available via progress endpoints
    
    **Input Options:**
    - **New Video File**: Upload video for complete processing pipeline
    - **Processed Video Hash**: Use hash from previous processing to skip preprocessing
    
    **Flutter EventSource Example:**
    ```dart
    // Start transcription
    final response = await http.post(
      Uri.parse('$baseUrl/transcribe/'),
      body: formData,
    );
    final taskId = json.decode(response.body)['task_id'];
    
    // Track progress with EventSource
    final eventSource = EventSource(Uri.parse('$baseUrl/progress/$taskId'));
    eventSource.listen((event) {
      final progress = json.decode(event.data);
      print('Progress: ${progress['percentage'].toStringAsFixed(1)}%');
      if (progress['status'] == 'completed') {
        final result = progress['result'];
        // Use the final transcription result
      }
    });
    ```
    
    **Python Example with New File:**
    ```python
    # Start transcription
    files = {'file': open('video.mp4', 'rb')}
    data = {'model_name': 'conformer', 'diacritized': True}
    response = requests.post('/transcribe/', files=files, data=data)
    task_id = response.json()['task_id']
    
    # Track progress
    import requests
    response = requests.get(f'/progress/{task_id}', stream=True)
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            progress_data = json.loads(line[6:])
            print(f"Progress: {progress_data['percentage']:.1f}%")
            if progress_data['status'] == 'completed':
                result = progress_data['result']
                break
    ```
    
    **Example Usage with Hash:**
    ```python
    data = {
        'file_hash': 'abc123def456789',
        'model_name': 'mstcn',
        'beam_size': 20
    }
    response = requests.post('/transcribe/', data=data)
    task_id = response.json()['task_id']
    ```
    """,
    responses={
        200: {"description": "Task started successfully, returns task_id for progress tracking"},
        400: {"description": "Invalid input parameters or unsupported file format"},
        413: {"description": "File too large"},
        500: {"description": "Internal server error"}
    }
)
async def transcribe_video(
    file: Optional[UploadFile] = File(
        None, 
        description="Video file for lip reading (MP4, AVI, MOV, MKV, WebM). Either this or file_hash must be provided."
    ),
    file_hash: Optional[str] = Form(
        None,
        description="Hash of a previously processed video file. Either this or file must be provided."
    ),
    model_name: str = Form(
        DEFAULT_MODEL, 
        description="Encoder model: 'mstcn' (fast), 'dctcn' (balanced), 'conformer' (most accurate)"
    ),
    landmark_model_name: str = Form(
        DEFAULT_LANDMARK_MODEL, 
        description="Landmark detection model: 'resnet50' (accurate) or 'mobilenet0.25' (fast)"
    ),
    diacritized: bool = Form(
        DEFAULT_DIACRITIZED, 
        description="Output diacritized Arabic text with vowel marks"
    ),
    enhance: bool = Form(
        False, 
        description="Apply AI enhancement using Google Gemini (requires API key)"
    ),
    beam_size: int = Form(
        DEFAULT_BEAM_SIZE, 
        description=f"Beam search width for decoding (1-{MAX_BEAM_SIZE}). Higher = more accurate but slower"
    ),
    include_summary: bool = Form(
        False, 
        description="Include AI-generated content summary (requires enhance=true)"
    ),
    include_translation: bool = Form(
        False, 
        description="Include translation to target language (requires enhance=true)"
    ),
    target_language: str = Form(
        "English", 
        description="Target language for translation (e.g., 'English', 'French', 'Spanish')"
    )
):
    """
    Start Arabic speech transcription from video using lip reading technology with progress tracking.
    
    This endpoint performs asynchronous end-to-end Arabic lip reading with the following pipeline:
    
    **Immediate Response:**
    - Returns a unique task_id for progress tracking
    - Processing starts immediately in the background
    - Client can track progress via Server-Sent Events (SSE)
    
    **Processing Pipeline:**
    1. **Input Validation**: Accept either a new video file or a hash of previously processed video
    2. **Video Processing** (if new file): Orientation detection, face detection, mouth extraction
    3. **Lip Reading Inference**: Apply deep learning model to predict Arabic text
    4. **AI Enhancement** (optional): Use Google Gemini for text improvement and additional features
    
    **Progress Tracking:**
    - Compatible with Flutter EventSource for real-time updates
    - Updates sent every second during processing
    - Final result available when status becomes 'completed'
    
    **Input Options:**
    - **New Video File**: Upload video for complete processing pipeline
    - **Processed Video Hash**: Use hash from previous processing to skip preprocessing
    
    **Model Performance:**
    - **MSTCN**: Fastest processing, good for real-time applications
    - **DenseTCN**: Balanced speed and accuracy
    - **Conformer**: Highest accuracy, recommended for best results
    
    **AI Enhancement Features:**
    - Text improvement and error correction
    - Content summarization
    - Translation to multiple languages
    - Contextual understanding
    
    **Flutter Integration Example:**
    ```dart
    // 1. Start transcription
    final response = await http.post(
      Uri.parse('$baseUrl/transcribe/'),
      body: formData,
    );
    final taskId = json.decode(response.body)['task_id'];
    
    // 2. Track progress with EventSource
    final eventSource = EventSource(Uri.parse('$baseUrl/progress/$taskId'));
    eventSource.listen((event) {
      final progress = json.decode(event.data);
      setState(() {
        progressPercentage = progress['percentage'];
        currentStep = progress['current_step'];
      });
      
      if (progress['status'] == 'completed') {
        final result = progress['result'];
        // Process final transcription result
        eventSource.close();
      } else if (progress['status'] == 'failed') {
        // Handle error
        eventSource.close();
      }
    });
    ```
    
    **Python Example with New File:**
    ```python
    import requests
    import json
    
    # Start transcription
    files = {'file': open('video.mp4', 'rb')}
    data = {
        'model_name': 'conformer',
        'diacritized': True,
        'enhance': True
    }
    response = requests.post('/transcribe/', files=files, data=data)
    task_id = response.json()['task_id']
    
    # Track progress
    response = requests.get(f'/progress/{task_id}', stream=True)
    for line in response.iter_lines():
        if line.startswith(b'data: '):
            progress_data = json.loads(line[6:])
            print(f"Progress: {progress_data['percentage']:.1f}%")
            if progress_data['status'] == 'completed':
                result = progress_data['result']
                # Response includes processed_video_hash for future use
                print(f"Hash for reuse: {final_result['video_hash']}")
                break
    ```
    
    **Example Usage with Hash:**
    ```python
    data = {
        'file_hash': 'abc123def456789',
        'model_name': 'mstcn',
        'beam_size': 20
    }
    response = requests.post('/transcribe/', data=data)
    task_id = response.json()['task_id']
    ```
    """
    # Validate input: either file or file_hash must be provided, but not both
    if file is None and file_hash is None:
        raise HTTPException(
            status_code=400, 
            detail="Either 'file' or 'file_hash' must be provided"
        )
    
    if file is not None and file_hash is not None:
        raise HTTPException(
            status_code=400, 
            detail="Cannot provide both 'file' and 'file_hash'. Choose one."
        )
    
    # Validate file_hash if provided
    if file_hash is not None and not is_valid_hash(file_hash):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file hash format. Hash must be a 16-character hexadecimal string."
        )
    # Validate beam size
    if not (1 <= beam_size <= MAX_BEAM_SIZE):
        raise HTTPException(
            status_code=400, 
            detail=f"Beam size must be between 1 and {MAX_BEAM_SIZE}"
        )
    
    saved_file_path = None
    video_file_path = None
    is_preprocessed = False
    
    try:
        # Log the start of processing
        logger.info(f"=== NEW TRANSCRIPTION REQUEST ===")
        
        # Determine input source and get video file path
        if file_hash:
            # Using hash - get processed file path
            logger.info(f"Using processed file hash: {file_hash}")
            video_file_path = get_file_from_hash(file_hash)
            is_preprocessed = True
            logger.info(f"Found processed file: {video_file_path}")
        else:
            # Using uploaded file - save and process
            logger.info(f"Processing new uploaded file: {file.filename}")
            saved_file_path = save_uploaded_file(file)
            video_file_path = saved_file_path
            is_preprocessed = False
            logger.info(f"Saved uploaded file: {video_file_path}")
        
        logger.info(f"Model: {model_name}, Landmark: {landmark_model_name}, Diacritized: {diacritized}, Beam size: {beam_size}, Preprocessed: {is_preprocessed}")
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Generated task ID: {task_id}")
        
        # Create progress tracker for this task
        progress_tracker.create_task(task_id, total_steps=100)
        
        # Start background processing
        asyncio.create_task(process_video_background(
            task_id=task_id,
            video_file_path=video_file_path,
            model_name=model_name,
            landmark_model_name=landmark_model_name,
            diacritized=diacritized,
            beam_size=beam_size,
            is_preprocessed=is_preprocessed,
            file_hash=file_hash,
            enhance=enhance,
            include_summary=include_summary,
            include_translation=include_translation,
            target_language=target_language
        ))
        
        logger.info(f"Background processing started for task: {task_id}")
        
        # Return task ID immediately for progress tracking
        return JSONResponse(content={
            "task_id": task_id,
            "message": "Transcription started. Use the task_id to track progress.",
            "progress_stream_url": f"/progress/{task_id}",
            "progress_status_url": f"/progress/{task_id}/status"
        })
        
    except HTTPException:
        logger.error("HTTP Exception occurred during transcription")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.info("=== REQUEST PROCESSING ENDED ===\n")
        flush_logs()  # Ensure all request logs are written

@app.post(
    "/enhance-text/", 
    response_model=TextEnhancementResponse,
    tags=["enhancement"],
    summary="Enhance Arabic text with AI",
    description="""
    Enhance Arabic text using Google Gemini AI without requiring video upload.
    
    **Features:**
    - Text improvement and error correction
    - Grammar and style enhancement
    - Content summarization
    - Multi-language translation
    - Contextual understanding
    
    **Use Cases:**
    - Improve transcription results from other sources
    - Translate Arabic text to other languages
    - Generate summaries of Arabic content
    - Polish and enhance Arabic writing
    """,
    responses={
        200: {"description": "Successfully enhanced text with AI"},
        503: {"description": "Gemini AI service not available"},
        500: {"description": "Text enhancement failed"}
    }
)
async def enhance_text(
    text: str = Form(..., description="Arabic text to enhance"),
    include_summary: bool = Form(
        False, 
        description="Generate a summary of the content"
    ),
    include_translation: bool = Form(
        False, 
        description="Translate the text to target language"
    ),
    target_language: str = Form(
        "English", 
        description="Target language for translation (e.g., 'English', 'French', 'Spanish')"
    )
):
    """
    Enhance Arabic text using Google Gemini AI.
    
    This endpoint provides AI-powered text enhancement capabilities without requiring
    video processing. It's useful for:
    
    - **Improving transcription quality** from other sources
    - **Translating Arabic text** to multiple languages
    - **Generating summaries** of Arabic content
    - **Grammar and style correction** for Arabic text
    
    **AI Capabilities:**
    - Advanced language understanding
    - Context-aware corrections
    - Natural translation preserving meaning
    - Intelligent summarization
    
    **Example Usage:**
    ```python
    data = {
        'text': 'مرحبا بكم في واجهة برمجة التطبيقات',
        'include_summary': True,
        'include_translation': True,
        'target_language': 'English'
    }
    response = requests.post('/enhance-text/', data=data)
    ```
    
    **Note:** Requires Google AI API key to be configured.
    """
    if not gemini_service:
        raise HTTPException(status_code=503, detail="Gemini service not available")
    
    try:
        # Apply AI enhancements
        enhancements = await _apply_ai_enhancements(text, include_summary, include_translation, target_language)
        
        response = {
            "original_text": text,
            "enhanced_text": enhancements.get("enhancement", {}).get("content", text),
            "enhancements": enhancements
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Text enhancement failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/config", 
    response_model=ConfigResponse,
    tags=["configuration"],
    summary="Get API configuration",
    description="""
    Retrieve current API configuration including:
    
    **Upload Settings:**
    - Maximum file size limits
    - Supported file formats
    
    **Processing Options:**
    - Available models and their capabilities
    - Default and maximum beam search sizes
    
    **AI Services:**
    - Availability of enhancement features
    
    This endpoint is useful for frontend applications to dynamically
    configure their UI based on current API capabilities.
    """,
    responses={
        200: {"description": "Current API configuration"}
    }
)
async def get_config():
    """
    Get current API configuration and capabilities.
    
    Returns comprehensive information about:
    - Upload limitations and supported formats
    - Available models and their characteristics
    - Processing parameters and defaults
    - AI service availability
    
    **Use this endpoint to:**
    - Configure frontend UI elements
    - Validate user inputs before submission
    - Display available options to users
    - Check AI service availability
    """
    max_size_mb = MAX_UPLOAD_SIZE / (1024 * 1024)
    
    # Available models and landmark models are hardcoded since they don't change
    available_models = ['mstcn', 'dctcn', 'conformer']
    available_landmark_models = ['resnet50', 'mobilenet0.25']
    
    return {
        "upload": {
            "max_size_mb": round(max_size_mb, 1),
            "max_size_bytes": MAX_UPLOAD_SIZE,
            "allowed_extensions": ALLOWED_EXTENSIONS
        },
        "processing": {
            "default_beam_size": DEFAULT_BEAM_SIZE,
            "max_beam_size": MAX_BEAM_SIZE
        },
        "model": {
            "default_model": DEFAULT_MODEL,
            "default_landmark_model": DEFAULT_LANDMARK_MODEL,
            "default_diacritized": DEFAULT_DIACRITIZED,
            "available_models": available_models,
            "available_landmark_models": available_landmark_models
        },
        "ai": {
            "gemini_available": gemini_service is not None
        }
    }

@app.get(
    "/", 
    response_model=APIInfo,
    tags=["configuration"],
    summary="API Information",
    description="""
    Get basic information about the Arabic Lip Reading API.
    
    **Provides:**
    - API version and features
    - Available endpoints overview
    - Quick start information
    
    **For detailed documentation:**
    - Visit `/docs` for interactive Swagger UI
    - Visit `/redoc` for ReDoc documentation
    - Use `/config` for current configuration
    """,
    responses={
        200: {"description": "API information and feature list"}
    }
)
async def root():
    """
    Get API information and overview.
    
    This endpoint provides a quick overview of the Arabic Lip Reading API,
    including its main features and available endpoints.
    
    **Main Features:**
    - Advanced Arabic lip reading with multiple model options
    - AI-powered text enhancement and translation
    - Automatic video processing and orientation correction
    - File hash caching for efficient re-processing
    - RESTful API with comprehensive documentation
    
    **Getting Started:**
    1. Check `/config` for current settings and available models
    2. Use `/transcribe/` to upload videos for lip reading (save the hash!)
    3. Use `/transcribe/` with hash for different models or parameters
    4. Use `/enhance-text/` for AI text enhancement
    5. Explore `/docs` for interactive API testing
    """
    return {
        "message": "Arabic Lip Reading API",
        "version": "1.0.0",
        "features": [
            "Multi-encoder support (MSTCN, DenseTCN, Conformer)",
            "Multiple landmark detection models (ResNet50, MobileNet)",
            "Diacritized and non-diacritized Arabic text output",
            "AI-powered text enhancement with Google Gemini",
            "Real-time progress tracking with Server-Sent Events",
            "Graceful task cancellation with resource cleanup",
            "Automatic model downloading from Google Drive",
            "Per-request model initialization for better isolation",
            "File hash caching for efficient video re-processing"
        ],
        "endpoints": {
            "transcribe": "/transcribe/ - Upload video or use hash for lip reading with model selection",
            "progress_stream": "/progress/{task_id} - Stream real-time progress updates via Server-Sent Events",
            "progress_status": "/progress/{task_id}/status - Get current progress status (one-time request)",
            "cancel_task": "/progress/{task_id}/cancel - Cancel a running transcription task (DELETE or POST)",
            "enhance": "/enhance-text/ - Enhance Arabic text with AI",
            "config": "/config - Get API configuration and available models",
            "api-docs": "/api-docs - Comprehensive API documentation with examples",
            "health": "/health - API health check and service status",
            "docs": "/docs - Interactive API documentation (Swagger UI)",
            "redoc": "/redoc - Alternative API documentation (ReDoc)"
        }
    }

@app.get(
    "/api-docs",
    tags=["configuration"],
    summary="Comprehensive API Documentation",
    description="Get detailed API documentation with examples and best practices",
    responses={
        200: {"description": "Detailed API documentation"}
    }
)
async def api_documentation():
    """
    Comprehensive API documentation with examples and best practices.
    """
    return {
        "title": "Arabic Lip Reading API - Complete Documentation",
        "version": "1.0.0",
        "description": "Advanced Arabic lip reading system with AI enhancement",
        
        "quick_start": {
            "description": "Get started with the API in 3 simple steps",
            "steps": [
                {
                    "step": 1,
                    "title": "Check Configuration",
                    "endpoint": "GET /config",
                    "description": "Get current API settings and available models"
                },
                {
                    "step": 2,
                    "title": "Upload Video",
                    "endpoint": "POST /transcribe/",
                    "description": """
                    Upload a video file for Arabic lip reading transcription.
                    
                    **Processing Pipeline:**
                    1. Video upload and validation
                    2. Face detection and landmark extraction  
                    3. Mouth region extraction and preprocessing
                    4. Lip reading model inference
                    5. Optional AI enhancement with Gemini
                    
                    **Returns a hash** for the processed video that can be used for future
                    transcriptions with different models or parameters without reprocessing.
                    """,
                    "parameters": {
                        "file": "Video file to process",
                        "model_name": "Encoder model (mstcn/dctcn/conformer)",
                        "landmark_model_name": "Landmark model (resnet50/mobilenet0.25)",
                        "enhance": "Apply AI enhancement (requires API key)"
                    },
                    "returns": {
                        "transcript": "Predicted Arabic text",
                        "hash": "Hash for processed video (for future use)",
                        "metadata": "Processing information and statistics"
                    }
                },
                {
                    "step": 2.5,
                    "title": "Reuse Processed Video (Optional)",
                    "endpoint": "POST /transcribe/",
                    "description": """
                    Reuse a previously processed video using its hash.
                    
                    **Benefits:**
                    - Skip video preprocessing (faster processing)
                    - Try different models or beam sizes
                    - Reduce computational overhead
                    - Consistent results across runs
                    """,
                    "parameters": {
                        "file_hash": "Hash from previous processing",
                        "model_name": "Different encoder model to try",
                        "beam_size": "Different beam search parameters"
                    }
                },
                {
                    "step": 3,
                    "title": "Enhance Results (Optional)",
                    "endpoint": "POST /enhance-text/",
                    "description": "Use AI to improve, summarize, or translate the results"
                }
            ]
        },
        
        "models": {
            "encoder_models": {
                "mstcn": {
                    "name": "Multi-Scale Temporal Convolutional Network",
                    "speed": "Fast",
                    "accuracy": "Good",
                    "best_for": "Real-time applications, quick processing",
                    "memory_usage": "Low"
                },
                "dctcn": {
                    "name": "Dense Temporal Convolutional Network",
                    "speed": "Medium",
                    "accuracy": "Better",
                    "best_for": "Balanced performance and accuracy",
                    "memory_usage": "Medium"
                },
                "conformer": {
                    "name": "Conformer (Transformer + Convolution)",
                    "speed": "Slower",
                    "accuracy": "Best",
                    "best_for": "Highest quality results, research applications",
                    "memory_usage": "High"
                }
            },
            "landmark_models": {
                "resnet50": {
                    "name": "ResNet-50",
                    "accuracy": "High",
                    "speed": "Medium",
                    "best_for": "Production use, high accuracy requirements"
                },
                "mobilenet0.25": {
                    "name": "MobileNet v1 0.25x",
                    "accuracy": "Good",
                    "speed": "Fast",
                    "best_for": "Mobile devices, edge computing"
                }
            }
        },
        
        "examples": {
            "basic_transcription": {
                "description": "Basic video transcription with default settings",
                "method": "POST",
                "endpoint": "/transcribe/",
                "curl_example": "curl -X POST 'http://localhost:8000/transcribe/' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'file=@video.mp4' -F 'model_name=conformer' -F 'diacritized=true'"
            },
            
            "enhanced_transcription": {
                "description": "Video transcription with AI enhancement and translation",
                "method": "POST",
                "endpoint": "/transcribe/",
                "curl_example": "curl -X POST 'http://localhost:8000/transcribe/' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'file=@video.mp4' -F 'model_name=conformer -F 'enhance=true' -F 'include_summary=true' -F 'include_translation=true' -F 'target_language=English'"
            },
            
            "hash_transcription": {
                "description": "Reuse processed video with hash for different model or parameters",
                "method": "POST",
                "endpoint": "/transcribe/",
                "curl_example": "curl -X POST 'http://localhost:8000/transcribe/' -H 'accept: application/json' -H 'Content-Type: application/x-www-form-urlencoded' -d 'file_hash=abc123def456789&model_name=mstcn&beam_size=20'"
            },
            
            "text_enhancement": {
                "description": "Enhance existing Arabic text with AI",
                "method": "POST",
                "endpoint": "/enhance-text/",
                "curl_example": "curl -X POST 'http://localhost:8000/enhance-text/' -H 'accept: application/json' -H 'Content-Type: application/x-www-form-urlencoded' -d 'text=مرحبا بكم في واجهة برمجة التطبيقات&include_translation=true&target_language=English'"
            }
        },
        
        "best_practices": {
            "video_quality": [
                "Use clear, well-lit videos for best results",
                "Ensure the speaker's face is clearly visible",
                "Avoid extreme angles or rotations",
                "Prefer videos with stable framing"
            ],
            "model_selection": [
                "Use 'conformer' for highest accuracy",
                "Use 'mstcn' for fastest processing",
                "Use 'dctcn' for balanced performance",
                "Consider memory constraints when choosing models"
            ],
            "ai_enhancement": [
                "Enable AI enhancement for production use",
                "Use summarization for long content",
                "Specify target language clearly for translation",
                "Ensure Google AI API key is configured"
            ],
            "performance": [
                "Process videos in batches for better throughput",
                "Use appropriate beam_size (10-20 for most cases)",
                "Monitor memory usage with larger models",
                "Cache results when possible",
                "Use file hashes to reuse processed videos",
                "Save hashes from initial processing for future use",
                "Skip preprocessing when using hash for faster inference"
            ]
        },
        
        "error_handling": {
            "common_errors": {
                "413": {
                    "description": "File too large",
                    "solution": "Reduce file size or check /config for limits"
                },
                "422": {
                    "description": "Video processing failed",
                    "solution": "Check video format and quality"
                },
                "503": {
                    "description": "Service unavailable",
                    "solution": "Check model availability or AI service configuration"
                }
            }
        },
        
        "rate_limits": {
            "description": "Currently no rate limits enforced",
            "recommendation": "Implement rate limiting in production environments"
        },
        
        "authentication": {
            "description": "Currently no authentication required",
            "recommendation": "Implement API key authentication for production use"
        }
    }

@app.get(
    "/health",
    tags=["configuration"],
    summary="API Health Check",
    description="Check API health and service availability",
    responses={
        200: {"description": "API is healthy"},
        503: {"description": "Some services unavailable"}
    }
)
async def health_check():
    """
    Check API health and service availability.
    
    This endpoint provides a quick health check for monitoring systems.
    """
    health_status = {
        "status": "healthy",
        "timestamp": "2025-06-28T00:00:00Z",
        "version": "1.0.0",
        "services": {
            "api": "healthy",
            "gemini_ai": "healthy" if gemini_service else "unavailable",
            "file_upload": "healthy",
            "logging": "healthy"
        },
        "configuration": {
            "max_upload_size_mb": round(MAX_UPLOAD_SIZE / (1024 * 1024), 1),
            "default_model": DEFAULT_MODEL,
            "ai_enhancement_available": gemini_service is not None
        }
    }
    
    # Determine overall status
    if not gemini_service:
        health_status["status"] = "degraded"
        return JSONResponse(content=health_status, status_code=503)
    
    return health_status