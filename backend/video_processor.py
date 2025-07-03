import os
import sys
import cv2
import torch
import gdown
import numpy as np
import logging
import subprocess
from pathlib import Path
from torchvision import transforms

# Add the model directory to the path to import our modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'model'))
sys.path.append(str(project_root / 'preparation' / 'retinaface'))

from preparation.retinaface.detector import LandmarksDetector
from preparation.retinaface.video_process import VideoProcess
from model.e2e_vsr import E2EVSR

# Constants for normalization (consistent across the pipeline)
INFERENCE_MEAN = 0.40947135433671134
INFERENCE_STD = 0.15003469454968454

# Initialize logger
logger = logging.getLogger(__name__)


class VideoPreprocessor:
    """
    Handles video preprocessing using RetinaFace for mouth region extraction with orientation correction.
    
    Key Features:
    - OpenCV-based video loading for better compatibility
    - Automatic orientation detection using face landmark voting
    - OpenCV-based video saving instead of torchvision
    - Integrated mouth region extraction pipeline
    - Support for various video formats and orientations
    
    Changes from original:
    - Added detect_orientation() method for automatic rotation correction
    - Replaced torchvision.io.read_video with OpenCV-based loading
    - Added save_video_opencv() method using OpenCV VideoWriter
    - Enhanced error handling and validation
    """
    
    def __init__(self, device, model_name='resnet50'):
        logger.info(f"Initializing VideoPreprocessor with device: {device}, model: {model_name}")
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")
        
        self.device = device
        # Download RetinaFace model if not already present
        if model_name == 'resnet50':
            model_path = project_root / 'preparation' / 'retinaface' / 'ibug' / 'face_detection' / 'retina_face' / 'weights' / 'Resnet50_Final.pth'
            if not model_path.exists():
                logger.info("RetinaFace model weights not found, downloading...")
                try:
                    # Ensure the directory exists
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    gdown.download('https://drive.google.com/uc?id=1iSBM7gVQABBSEFw0RQQr9hcKYBNqi8HI', str(model_path), quiet=False)
                    logger.info("RetinaFace model weights downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download RetinaFace model weights: {e}")
                    raise
            else:
                logger.info("RetinaFace model weights already exist, skipping download")
        
        try:
            self.landmarks_detector = LandmarksDetector(device=device, model_name=model_name)
            logger.info("LandmarksDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LandmarksDetector: {e}")
            raise
        
        try:
            self.video_process = VideoProcess(convert_gray=True)
            logger.info("VideoProcess initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VideoProcess: {e}")
            raise
        
        logger.info("VideoPreprocessor initialization completed")
    
    def load_video(self, video_path):
        """Load video from file path using OpenCV."""
        logger.info(f"Loading video from: {video_path}")
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"cv2.VideoCapture failed to open video: {video_path}")
                raise RuntimeError(f"cv2.VideoCapture failed to open video '{video_path}'")

            frames_lst = []
            frame_count = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames_lst.append(rgb)
                frame_count += 1
            cap.release()

            if len(frames_lst) == 0:
                logger.error(f"No frames read from video: {video_path}")
                raise RuntimeError(f"No frames read from '{video_path}' via cv2.VideoCapture")

            frames = np.stack(frames_lst, axis=0).astype(np.uint8)  # (T,H,W,C)
            logger.info(f"Successfully loaded {frame_count} frames from video. Shape: {frames.shape}")
            return frames
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            raise
    
    def detect_orientation_metadata(self, video_path):
        """
        Detect video orientation using ffprobe metadata (fast and accurate).
        
        This is the optimized approach that reads rotation metadata directly from
        the video file instead of testing landmark detection on different orientations.
        
        Returns:
            tuple: (rotation_degrees, rotation_function)
                - rotation_degrees: 0, 90, 180, or 270
                - rotation_function: OpenCV rotation function to apply
        """
        logger.info(f"Detecting orientation using metadata for: {video_path}")
        
        def get_rotation_from_metadata(path: str) -> int:
            """Return rotation degrees (0/90/180/270) from video metadata using ffprobe."""
            
            # Check if ffprobe is available
            try:
                result = subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5)
                if result.returncode != 0:
                    logger.debug("ffprobe not available or not working")
                    return 0
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logger.debug(f"ffprobe not available: {e}")
                return 0
            
            # Try side_data first (works for 180°)
            cmd_side = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream_side_data=rotation",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ]
            try:
                result = subprocess.run(cmd_side, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    rotation = abs(int(float(result.stdout.strip()))) % 360
                    logger.debug(f"Found rotation from side_data: {rotation}°")
                    return rotation
            except Exception as e:
                logger.debug(f"Side_data rotation check failed: {e}")
            
            # Try tag variant (works for 90°/270°)
            cmd_tag = [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream_tags=rotate",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ]
            try:
                result = subprocess.run(cmd_tag, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    rotation = abs(int(float(result.stdout.strip()))) % 360
                    logger.debug(f"Found rotation from tags: {rotation}°")
                    return rotation
            except Exception as e:
                logger.debug(f"Tag rotation check failed: {e}")
            
            logger.debug("No rotation metadata found")
            return 0
        
        try:
            rotation_degrees = get_rotation_from_metadata(video_path)
            logger.info(f"Metadata-based orientation detection result: {rotation_degrees}°")
            
            # Map rotation degrees to OpenCV rotation flags
            rotation_map = {
                0: (None, lambda fr: fr),
                90: (cv2.ROTATE_90_CLOCKWISE, lambda fr: cv2.rotate(fr, cv2.ROTATE_90_CLOCKWISE)),
                180: (cv2.ROTATE_180, lambda fr: cv2.rotate(fr, cv2.ROTATE_180)),
                270: (cv2.ROTATE_90_COUNTERCLOCKWISE, lambda fr: cv2.rotate(fr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            }
            
            rotation_flag, rotation_fn = rotation_map.get(rotation_degrees, (None, lambda fr: fr))
            
            if rotation_degrees != 0:
                logger.info(f"Will apply rotation: {rotation_degrees}° ({rotation_flag})")
            else:
                logger.info("No rotation correction needed")
                
            return rotation_degrees, rotation_fn
            
        except Exception as e:
            logger.warning(f"Metadata-based orientation detection failed: {e}")
            logger.info("Falling back to landmark-based orientation detection")
            return self.detect_orientation_fallback(video_path)

    def save_video(self, filename, vid, frames_per_second):
        """Save video using OpenCV."""
        logger.info(f"Saving video to: {filename} at {frames_per_second} FPS")
        logger.debug(f"Video data shape: {vid.shape}, dtype: {vid.dtype}")
        
        try:
            # Use pathlib for directory creation
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert from tensor to numpy if needed
            if torch.is_tensor(vid):
                vid = vid.numpy()
                logger.debug("Converted tensor to numpy array")
            
            # Ensure vid is in the right format (T, H, W) or (T, H, W, C)
            if vid.ndim == 3:  # (T, H, W) - grayscale
                height, width = vid.shape[1], vid.shape[2]
                # Convert grayscale to RGB for video writing
                vid_rgb = np.stack([vid, vid, vid], axis=-1)  # (T, H, W, 3)
                logger.debug(f"Converted grayscale to RGB: {vid_rgb.shape}")
            elif vid.ndim == 4 and vid.shape[-1] == 1:  # (T, H, W, 1)
                vid_rgb = np.repeat(vid, 3, axis=-1)  # (T, H, W, 3)
                height, width = vid.shape[1], vid.shape[2]
                logger.debug(f"Converted single-channel to RGB: {vid_rgb.shape}")
            elif vid.ndim == 4 and vid.shape[-1] == 3:  # (T, H, W, 3)
                vid_rgb = vid
                height, width = vid.shape[1], vid.shape[2]
                logger.debug(f"Using existing RGB format: {vid_rgb.shape}")
            else:
                logger.error(f"Unsupported video format: shape={vid.shape}")
                raise ValueError(f"Unsupported video format: {vid.shape}")
        
        except Exception as e:
            logger.error(f"Error preparing video data for saving: {e}")
            raise
        
        # Ensure data is uint8
        if vid_rgb.dtype != np.uint8:
            logger.debug(f"Converting data type from {vid_rgb.dtype} to uint8")
            vid_rgb = np.clip(vid_rgb, 0, 255).astype(np.uint8)
        
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, frames_per_second, (width, height))
            
            if not out.isOpened():
                logger.error(f"Failed to open video writer for: {filename}")
                raise RuntimeError(f"Failed to create video writer for {filename}")
            
            # Write frames
            frames_written = 0
            for frame in vid_rgb:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                frames_written += 1
            
            out.release()
            logger.info(f"Successfully saved {frames_written} frames to {filename}")
            
        except Exception as e:
            logger.error(f"Error writing video to {filename}: {e}")
            raise

    def extract_mouth_region(self, video_path):
        """
        Extract mouth region from video using RetinaFace landmarks detection.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Tuple of (processed_video_tensor, output_path):
            - output_path: Path to the saved processed video file
        """
        logger.info(f"Starting mouth region extraction for: {video_path}")
        
        try:
            # Apply orientation correction first (before loading all frames)
            try:
                logger.info("Starting orientation correction using metadata")
                rotation_degrees, rotation_fn = self.detect_orientation_metadata(video_path)
            except Exception as e:
                logger.warning(f"Orientation detection failed – proceeding without rotation: {e}")
                rotation_degrees, rotation_fn = 0, lambda fr: fr
            
            # Load video
            video = self.load_video(video_path)
            logger.info(f"Loaded video successfully, shape: {video.shape}")

            # Apply rotation if needed
            if rotation_degrees != 0:
                logger.info(f"Applying orientation correction: {rotation_degrees}°")
                video = np.stack([rotation_fn(fr) for fr in video])
                logger.info(f"Orientation correction applied, new shape: {video.shape}")
            else:
                logger.info("No orientation correction needed")
            
            # Optional global down-resize for faster landmark detection (optimization from inference_vsr.ipynb)
            target_size = (256, 256)
            if video.shape[1] != target_size[1] or video.shape[2] != target_size[0]:
                logger.info(f"Resizing frames from {video.shape[1]}x{video.shape[2]} to {target_size[1]}x{target_size[0]} for faster processing")
                video = np.array([
                    cv2.resize(fr, target_size, interpolation=cv2.INTER_AREA) for fr in video
                ])
                logger.info(f"Global resize completed, new shape: {video.shape}")
            
            # Detect landmarks
            logger.info("Starting landmark detection")
            landmarks = self.landmarks_detector(video)
            valid_landmarks = sum(1 for lm in landmarks if lm is not None)
            logger.info(f"Landmark detection completed: {valid_landmarks}/{len(landmarks)} frames have valid landmarks")
            
            if valid_landmarks == 0:
                logger.error("No landmarks detected in any frame")
                raise RuntimeError("No landmarks detected in video")
            
            # Process video with landmarks to extract mouth region
            logger.info("Starting mouth region extraction")
            processed_video = self.video_process(video, landmarks)
            
            if processed_video is None:
                logger.error("VideoProcess returned None")
                raise RuntimeError("VideoProcess returned None — unable to generate mouth ROI sequence.")
            
            logger.info(f"Mouth region extraction completed, processed video shape: {processed_video.shape}")
            
            # Convert to tensor
            video_tensor = torch.tensor(processed_video, dtype=torch.float32)
            logger.debug(f"Converted to tensor: {video_tensor.shape}, dtype: {video_tensor.dtype}")

            # Save to video file using OpenCV for caching/debugging
            # Use backend's configured processed directory
            backend_dir = Path(__file__).parent
            processed_dir = backend_dir / 'processed'
            processed_dir.mkdir(parents=True, exist_ok=True)
            output_path = processed_dir / Path(video_path).name
            logger.info(f"Saving processed video to: {output_path}")
            self.save_video(str(output_path), video_tensor, frames_per_second=30)
            
            logger.info("Mouth region extraction completed successfully")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to extract mouth region from {video_path}: {e}")
            raise RuntimeError(f"Failed to preprocess video: {str(e)}")

    def detect_orientation_fallback(self, video_path):
        """
        Fallback orientation detection using landmark voting (for when ffprobe fails).
        
        This is the original method that loads frames and tests landmark detection
        on different orientations to find the best one.
        """
        logger.info("Starting fallback orientation detection using landmark voting")
        
        try:
            # Load a subset of frames for testing
            frames = self.load_video(video_path)
            logger.debug(f"Loaded frames for orientation testing: {frames.shape}")
            
            # Use up to 5 evenly-spaced frames to keep the detector overhead negligible
            num_probe = min(5, len(frames))
            probe_idx = np.linspace(0, len(frames) - 1, num=num_probe, dtype=int)
            probe_subset = frames[probe_idx]
            logger.debug(f"Using {num_probe} probe frames for orientation detection")

            # Define the four candidate orientations
            rotations = {
                None: lambda fr: fr,                                                                        # 0°
                cv2.ROTATE_90_COUNTERCLOCKWISE: lambda fr: cv2.rotate(fr, cv2.ROTATE_90_COUNTERCLOCKWISE),  # 90° CCW
                cv2.ROTATE_90_CLOCKWISE: lambda fr: cv2.rotate(fr, cv2.ROTATE_90_CLOCKWISE),                # 90° CW
                cv2.ROTATE_180: lambda fr: cv2.rotate(fr, cv2.ROTATE_180),                                  # 180°
            }

            hit_table = {}
            for rot_flag, fn in rotations.items():
                try:
                    test_frames = np.stack([fn(fr) for fr in probe_subset])
                    lm = self.landmarks_detector(test_frames)
                    hit_count = sum(l is not None for l in lm)
                    hit_table[rot_flag] = hit_count
                    logger.debug(f"Rotation {rot_flag}: {hit_count} landmarks detected")
                except Exception as e:
                    logger.warning(f"Error testing rotation {rot_flag}: {e}")
                    hit_table[rot_flag] = 0

            # Pick orientation(s) with the maximum hit count
            max_hits = max(hit_table.values()) if hit_table else 0
            candidates = [r for r, h in hit_table.items() if h == max_hits and h > 0]
            
            logger.info(f"Fallback orientation detection results: {hit_table}")
            logger.info(f"Max hits: {max_hits}, Candidates: {candidates}")

            best_rot = None
            if candidates:
                if len(candidates) == 1:
                    best_rot = candidates[0]
                    logger.info(f"Single best orientation found: {best_rot}")
                else:
                    # Tie-break: if the raw buffer is landscape (W>H) prefer
                    # a 90-degree rotation that turns it portrait; otherwise
                    # keep the original orientation.
                    h, w = frames.shape[1:3]
                    want_swap = w > h
                    logger.debug(f"Tie-breaking: frame dimensions {w}x{h}, want_swap: {want_swap}")
                    for cand in candidates:
                        swap_dims = cand in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        if swap_dims == want_swap:
                            best_rot = cand
                            logger.info(f"Tie-break winner: {best_rot}")
                            break
                    # Fallback to first candidate
                    if best_rot is None:
                        best_rot = candidates[0]
                        logger.info(f"Fallback to first candidate: {best_rot}")

            # If *no* orientation produced landmarks fall back to simple width>height heuristic
            if best_rot is None and frames.shape[2] > frames.shape[1]:
                best_rot = cv2.ROTATE_90_COUNTERCLOCKWISE
                logger.info(f"No landmarks detected, using heuristic rotation: {best_rot}")
            elif best_rot is None:
                logger.warning("No optimal orientation detected, keeping original")

            logger.info(f"Final fallback orientation decision: {best_rot}")
            
            # Convert to degrees for consistency with metadata method
            rotation_to_degrees = {
                None: 0,
                cv2.ROTATE_90_CLOCKWISE: 90,
                cv2.ROTATE_180: 180,
                cv2.ROTATE_90_COUNTERCLOCKWISE: 270,
            }
            
            degrees = rotation_to_degrees.get(best_rot, 0)
            rotation_fn = rotations.get(best_rot, lambda fr: fr)
            
            return degrees, rotation_fn
            
        except Exception as e:
            logger.error(f"Fallback orientation detection failed: {e}")
            logger.warning("Using no rotation as final fallback")
            return 0, lambda fr: fr

class LipReadingPredictor:
    """Handles lip reading model inference."""
    
    def __init__(self, model_name='conformer', dia=True, device=None):
        logger.info(f"Initializing LipReadingPredictor with model: {model_name}, diacritized: {dia}")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto-detected device: {device}")
        
        self.device = device
        self.model = None
        self.model_name = model_name
        self.dia = dia
        
        logger.info("Building token list")
        self.token_list = self._build_token_list(dia)
        self.vocab_size = len(self.token_list)
        logger.info(f"Token list built: {self.vocab_size} tokens")
        
        try:
            self.model_downloader = ModelDownloader()
            logger.info("ModelDownloader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ModelDownloader: {e}")
            raise

    def _build_token_list(self, dia):
        """Build token list for Arabic characters based on your mapping."""
        # This should match your mapped_tokens from master.py
        # You might want to load this from a config file or pass it as parameter
        if dia:
            tokens = ['', 'ٱ', 'يْ', 'يّْ', 'يِّ', 'يُّ', 'يَّ', 'يٌّ', 'يِ', 'يُ', 'يَ', 'يٌ', 'ي', 'ى', 'وْ', 'وِّ', 'وُّ', 'وَّ', 'وِ', 'وُ', 'وَ', 'وً', 'و', 'هْ', 'هُّ', 'هِ', 'هُ', 'هَ', 'نۢ', 'نْ', 'نِّ', 'نُّ', 'نَّ', 'نِ', 'نُ', 'نَ', 'مْ', 'مّْ', 'مِّ', 'مُّ', 'مَّ', 'مِ', 'مُ', 'مَ', 'مٍ', 'مٌ', 'مً', 'لْ', 'لّْ', 'لِّ', 'لُّ', 'لَّ', 'لِ', 'لُ', 'لَ', 'لٍ', 'لٌ', 'لً', 'كْ', 'كِّ', 'كَّ', 'كِ', 'كُ', 'كَ', 'قْ', 'قَّ', 'قِ', 'قُ', 'قَ', 'قٍ', 'قً', 'فْ', 'فِّ', 'فَّ', 'فِ', 'فُ', 'فَ', 'غْ', 'غِ', 'غَ', 'عْ', 'عَّ', 'عِ', 'عُ', 'عَ', 'عٍ', 'ظْ', 'ظِّ', 'ظَّ', 'ظِ', 'ظُ', 'ظَ', 'طْ', 'طِّ', 'طَّ', 'طِ', 'طُ', 'طَ', 'ضْ', 'ضِّ', 'ضُّ', 'ضَّ', 'ضِ', 'ضُ', 'ضَ', 'ضً', 'صْ', 'صّْ', 'صِّ', 'صُّ', 'صَّ', 'صِ', 'صُ', 'صَ', 'صٍ', 'صً', 'شْ', 'شِّ', 'شُّ', 'شَّ', 'شِ', 'شُ', 'شَ', 'سْ', 'سّْ', 'سِّ', 'سُّ', 'سَّ', 'سِ', 'سُ', 'سَ', 'سٍ', 'زْ', 'زَّ', 'زِ', 'زُ', 'زَ', 'رْ', 'رِّ', 'رُّ', 'رَّ', 'رِ', 'رُ', 'رَ', 'رٍ', 'رٌ', 'رً', 'ذْ', 'ذَّ', 'ذِ', 'ذُ', 'ذَ', 'دْ', 'دِّ', 'دُّ', 'دَّ', 'دًّ', 'دِ', 'دُ', 'دَ', 'دٍ', 'دٌ', 'دً', 'خْ', 'خِ', 'خُ', 'خَ', 'حْ', 'حَّ', 'حِ', 'حُ', 'حَ', 'جْ', 'جِّ', 'جُّ', 'جَّ', 'جِ', 'جُ', 'جَ', 'ثْ', 'ثِّ', 'ثُّ', 'ثَّ', 'ثِ', 'ثُ', 'ثَ', 'تْ', 'تِّ', 'تُّ', 'تَّ', 'تِ', 'تُ', 'تَ', 'تٍ', 'تٌ', 'ةْ', 'ةِ', 'ةُ', 'ةَ', 'ةٍ', 'ةٌ', 'ةً', 'بْ', 'بِّ', 'بَّ', 'بِ', 'بُ', 'بَ', 'بٍ', 'بً', 'ا', 'ئْ', 'ئِ', 'ئَ', 'ئً', 'إِ', 'ؤْ', 'ؤُ', 'ؤَ', 'أْ', 'أُ', 'أَ', 'آ', 'ءْ', 'ءِ', 'ءَ', 'ءً', ' ', '<sos>', '<eos>']
        else:
            tokens = ['', 'ٱ', 'ي', 'ى', 'و', 'ه', 'ن', 'م', 'ل', 'ك', 'ق', 'ف', 'غ', 'ع', 'ظ', 'ط', 'ض', 'ص', 'ش', 'س', 'ز', 'ر', 'ذ', 'د', 'خ', 'ح', 'ج', 'ث', 'ت', 'ة', 'ب', 'ا', 'ئ', 'إ', 'ؤ', 'أ', 'آ', 'ء', ' ', '<sos>', '<eos>']
        return tokens
    
    def load_model(self, model_name=None):
        """Load the trained E2EVSR model for the specified encoder type with caching optimization."""
        if model_name is None:
            model_name = self.model_name
            
        try:
            # Download model if needed
            model_path = self.model_downloader.download_model(model_name, dia=self.dia)
            
            # Check for cached model (following inference_vsr.ipynb optimization)
            model_path_obj = Path(model_path)
            cached_model_path = model_path_obj.parent / (model_path_obj.stem + '_cached' + model_path_obj.suffix)
            
            if cached_model_path.exists():
                logger.info(f"Loading cached model from: {cached_model_path}")
                try:
                    self.model = torch.load(str(cached_model_path), map_location="cpu", weights_only=False)
                    logger.info("Cached model loaded successfully!")
                    
                    logger.info(f"Moving cached model to device: {self.device}")
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info("Cached model ready for inference")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}. Building from checkpoint...")
            
            # Build model from scratch (first time or cache failed)
            logger.info("Building model from checkpoint...")
            
            # Get model configuration
            model_config = self.model_downloader.get_model_config(model_name)
            encoder_type = model_config['encoder_type']
            enc_options = {
                **model_config['enc_options'],
                'hidden_dim': model_config['enc_options'][f'{encoder_type}_options']['hidden_dim'],
                'frontend3d_dropout_rate': 0.0,
                'resnet_dropout_rate': 0.0
            }
            
            # Standard decoder options
            dec_options = {
                'attention_dim': 768,
                'attention_heads': 12,
                'linear_units': 3072,
                'num_blocks': 6,
                'dropout_rate': 0.1,
                'positional_dropout_rate': 0.1,
                'self_attention_dropout_rate': 0.1,
                'src_attention_dropout_rate': 0.1,
                'normalize_before': True,
            }
            
            # Create model
            self.model = E2EVSR(
                encoder_type=encoder_type,
                vocab_size=self.vocab_size,
                token_list=self.token_list,
                sos=self.vocab_size - 2,  # <sos> token
                eos=self.vocab_size - 1,  # <eos> token
                pad=0,  # blank token
                enc_options=enc_options,
                dec_options=dec_options,
                ctc_weight=0.5,
                label_smoothing=0.1,
            )
            logger.info("E2EVSR model created successfully")
            
            # Load the trained weights
            logger.info(f"Loading trained weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Check if this is a dummy model
            if isinstance(checkpoint, dict) and checkpoint.get('dummy', False):
                logger.warning(f"Using dummy model for testing: {checkpoint}")
            else:
                # Load real trained weights
                logger.info("Loading real trained weights")
                self.model.load_state_dict(checkpoint)
                logger.info(f"Model {model_name} loaded successfully!")
            
            logger.info(f"Moving model to device: {self.device}")
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model set to evaluation mode")
            
            # Save cached model for future runs (optimization)
            try:
                logger.info(f"Saving cached model to: {cached_model_path}")
                torch.save(self.model.cpu(), str(cached_model_path))
                logger.info("Cached model saved successfully for future runs")
                
                # Move model back to target device
                self.model.to(self.device)
                
            except Exception as e:
                logger.warning(f"Failed to save cached model: {e}. This won't affect functionality.")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")
    
    def predict(self, video_tensor, beam_size):
        """
        Perform lip reading prediction on preprocessed video tensor.
        
        Args:
            video_tensor: Preprocessed video tensor from VideoPreprocessor
            beam_size: Beam search size for decoding
            
        Returns:
            Predicted Arabic text
        """
        logger.info(f"Starting prediction with beam_size={beam_size}")
        logger.debug(f"Input tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        
        if self.model is None:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded. Please load a model first.")
        
        try:
            with torch.no_grad():
                # Add batch dimension if needed
                if video_tensor.dim() == 4:  # [C, T, H, W]
                    video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]
                    logger.debug(f"Added batch dimension: {video_tensor.shape}")

                # Move to device
                video_tensor = video_tensor.to(self.device)
                logger.debug(f"Moved tensor to device: {self.device}")
                
                # Get sequence length
                x_lengths = torch.tensor([video_tensor.size(2)], device=self.device)
                logger.debug(f"Sequence length: {x_lengths.item()}")

                # Perform beam search decoding
                logger.info("Starting beam search decoding")
                predictions = self.model.beam_search(
                    video_tensor, 
                    x_lengths, 
                    beam_size=beam_size, 
                    ctc_weight=0.3
                )
                logger.info("Beam search completed")
                
                # Convert predictions to text
                if predictions and len(predictions) > 0:
                    # Get the best hypothesis
                    predicted_ids = predictions[0]
                    logger.debug(f"Predicted IDs: {predicted_ids}")
                    
                    # Convert IDs to text
                    predicted_text = self._ids_to_text(predicted_ids)
                    logger.info(f"Prediction completed: '{predicted_text}'")
                    return predicted_text
                else:
                    logger.warning("No predictions returned from beam search")
                    return ""
                    
        except Exception as e:
            logger.error(f"Failed to perform prediction: {e}")
            raise RuntimeError(f"Failed to perform prediction: {str(e)}")
    
    def _ids_to_text(self, ids):
        """Convert token IDs to Arabic text."""
        logger.debug(f"Converting IDs to text: {ids}")
        text_tokens = []
        for token_id in ids:
            if 0 <= token_id < len(self.token_list):
                token = self.token_list[token_id]
                if token not in ['<sos>', '<eos>']:
                    text_tokens.append(token)
        
        return ''.join(text_tokens).strip()

class VideoInferenceService:
    """Complete service for video lip reading inference."""

    def __init__(self, model_name='conformer', landmark_model_name='resnet50', dia=True, device=None):
        """
        Initialize VideoInferenceService.
        
        Args:
            model_name: Name of the encoder model ('mstcn', 'dctcn', 'conformer')
            landmark_model_name: Name of the landmark detection model ('resnet50', etc.)
            dia: Whether to use diacritized Arabic text
            device: Device to run inference on
        """
        logger.info(f"Initializing VideoInferenceService with model={model_name}, landmark_model={landmark_model_name}, dia={dia}")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.landmark_model_name = landmark_model_name
        self.dia = dia
        
        try:
            logger.info("Initializing VideoPreprocessor")
            self.preprocessor = VideoPreprocessor(device=self.device, model_name=landmark_model_name)
            logger.info("VideoPreprocessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VideoPreprocessor: {e}")
            raise
        
        try:
            logger.info("Initializing LipReadingPredictor")
            self.predictor = LipReadingPredictor(model_name=model_name, dia=dia, device=self.device)
            logger.info("LipReadingPredictor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LipReadingPredictor: {e}")
            raise
        
        # Load the model
        try:
            logger.info("Loading model")
            self.predictor.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        logger.info("VideoInferenceService initialization completed")

    def load_video(self, video_path):
        """
        Load video from file path.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            Loaded video tensor
        """
        logger.info(f"Loading and preprocessing video: {video_path}")
        
        try:
            frames = self.preprocessor.load_video(video_path)
            if len(frames) == 0:
                logger.error("No frames to process for inference")
                raise RuntimeError("No frames to process for inference")

            logger.info(f"Loaded {len(frames)} frames")

            # Apply center cropping to 88x88 using transforms
            logger.debug("Applying transformations: grayscale, resize, center crop")
            crop_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((96, 96), antialias=True),  # Resize to 96x96 first
                transforms.CenterCrop((88, 88))  # Then center crop to 88x88
            ])
            
            frame_tensors = []
            for i, img in enumerate(frames):
                tensor = crop_transform(img)
                frame_tensors.append(tensor)
                if i % 100 == 0:  # Log every 100 frames
                    logger.debug(f"Processed {i+1}/{len(frames)} frames")
        
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            raise
        
        video = torch.stack(frame_tensors, dim=0)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)          # (C, T, H, W)
        video = transforms.Normalize(mean=[INFERENCE_MEAN], std=[INFERENCE_STD])(video)
        
        logger.debug(f"Final video tensor shape: {video.shape}")
        logger.info("Video loading and preprocessing completed")

        return video.unsqueeze(0)  # Add batch dimension: (1, C, T, H, W)

    def process_video(self, video_path, beam_size=10, is_preprocessed=False, cancellation_callback=None):
        """
        Complete pipeline: preprocess video and perform lip reading prediction.
        
        Args:
            video_path: Path to input video file
            beam_size: Beam search size for decoding
            is_preprocessed: Whether the video is already processed (mouth region extracted)
            cancellation_callback: Optional function that returns True if processing should be cancelled
            
        Returns:
            Tuple of (predicted_text, metadata)
        """
        logger.info(f"Starting video processing for: {video_path} with beam_size={beam_size}, is_preprocessed={is_preprocessed}")
        
        metadata = {
            'model_name': self.model_name,
            'diacritized': self.dia,
            'beam_size': beam_size,
            'success': False,
            'error': None
        }
        
        def check_cancellation():
            """Helper function to check for cancellation."""
            if cancellation_callback and cancellation_callback():
                raise RuntimeError("Task was cancelled")
        
        try:
            # Check for cancellation before preprocessing
            check_cancellation()
            
            if is_preprocessed:
                # Video is already processed (mouth region extracted), load directly
                logger.info("Loading already processed video for inference")
                video_tensor = self.load_video(video_path)
            else:
                # Video needs preprocessing - extract mouth region first
                logger.info("Starting mouth region extraction")
                check_cancellation()  # Check before preprocessing
                processed_video_path = self.preprocessor.extract_mouth_region(video_path)
                logger.info(f"Mouth region extraction completed: {processed_video_path}")
                
                logger.info("Loading processed video for inference")
                check_cancellation()  # Check before loading
                video_tensor = self.load_video(processed_video_path)

            # Check for cancellation before prediction
            check_cancellation()
            
            # Perform prediction using the processed video tensor
            predicted_text = self.predictor.predict(video_tensor, beam_size=beam_size)
            metadata['success'] = True
            
            return predicted_text, metadata
            
        except Exception as e:
            metadata['error'] = str(e)
            return "", metadata
    
    def get_available_models(self):
        """Get list of available encoder model names."""
        return self.predictor.model_downloader.list_available_models()
    
    def get_available_landmark_models(self):
        """Get list of available landmark detection model names."""
        return ['resnet50', 'mobilenet0.25']  # Common RetinaFace models


class ModelDownloader:
    """Handles downloading models from Google Drive using gdown."""
    
    def __init__(self):
        """Initialize ModelDownloader."""
        logger.info("Initializing ModelDownloader")
        self.models_dir = Path(__file__).parent / 'models'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Models directory: {self.models_dir}")
    
    # Model configurations for different encoders
    MODEL_CONFIGS = {
        'mstcn': {
            'filename_dia': 'mstcn_model_dia.pth',
            'filename_nodia': 'mstcn_model_nodia.pth',
            'encoder_type': 'mstcn',  # Fixed: should be 'mstcn' not 'multiscaletcn'
            'enc_options': {
                'mstcn_options': {
                    'tcn_type': 'multiscale',
                    'hidden_dim': 768,
                    'num_channels': [512, 512, 512, 512],
                    'kernel_size': [3, 5, 7, 9],                   
                    'dropout': 0.2,
                    'stride': 1,
                    'width_mult': 1.0,
                }
            }
        },
        'dctcn': {
            'filename_dia': 'dctcn_model_dia.pth',
            'filename_nodia': 'dctcn_model_nodia.pth',
            'encoder_type': 'densetcn',
            'enc_options': {
                'densetcn_options': {
                    'block_config': [4, 4, 4, 4],
                    'growth_rate_set': [512, 512, 512, 512],
                    'reduced_size': 768,
                    'kernel_size_set': [3, 5, 7, 9],
                    'dilation_size_set': [1, 2, 4, 8],
                    'squeeze_excitation': True,
                    'dropout': 0.2,
                    'hidden_dim': 768,
                }
            }
        },
        'conformer': {
            'filename_dia': 'conformer_model_dia.pth',
            'filename_nodia': 'conformer_model_nodia.pth',
            'encoder_type': 'conformer',
            'enc_options': {
                'conformer_options': {
                    'attention_dim': 768,
                    'attention_heads': 12,
                    'linear_units': 3072,
                    'num_blocks': 12,
                    'dropout_rate': 0.1,
                    'positional_dropout_rate': 0.1,
                    'attention_dropout_rate': 0.1,
                    'cnn_module_kernel': 31,
                    'hidden_dim': 768
                }
            }
        }
    }
    
    def download_model(self, model_name, dia=True):
        """
        Download a specific model if it doesn't exist locally.
        
        Args:
            model_name: One of 'mstcn', 'dctcn', 'conformer'
            dia: Whether to download diacritized version
            
        Returns:
            Path to the downloaded model file
        """
        logger.info(f"Requesting model download: {model_name}, diacritized={dia}")
        
        if model_name not in self.MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_CONFIGS.keys())}")
        
        config = self.MODEL_CONFIGS[model_name]
        # Ensure models_dir is a Path object for proper path operations
        models_dir_path = Path(self.models_dir) if isinstance(self.models_dir, str) else self.models_dir
        
        # Select appropriate filename based on diacritization setting
        filename_key = 'filename_dia' if dia else 'filename_nodia'
        model_filename = config[filename_key]
        model_path = models_dir_path / model_filename
        logger.debug(f"Model path: {model_path}")
        
        # Check if model already exists
        if model_path.exists():
            logger.info(f"Model {model_name} ({'dia' if dia else 'nodia'}) already exists at {model_path}")
            return str(model_path)
        
        # Get download URL from environment
        env_key = f'{model_name.upper()}_MODEL_{"DIA" if dia else "NODIA"}'
        download_url = os.getenv(env_key)
        logger.debug(f"Environment key: {env_key}, URL found: {bool(download_url)}")
        
        if not download_url:
            logger.warning(f"No download URL found for {model_name} ({'dia' if dia else 'nodia'})")
            return self._create_dummy_model(str(model_path), model_name, "No download URL found")
        
        # Download the model
        try:
            logger.info(f"Downloading {model_name} ({'dia' if dia else 'nodia'}) model from {download_url}")
            gdown.download(download_url, str(model_path), quiet=False)
            
            if model_path.exists():
                logger.info(f"Successfully downloaded {model_name} ({'dia' if dia else 'nodia'}) model")
                return str(model_path)
            else:
                logger.error(f"Download completed but file not found: {model_path}")
                raise RuntimeError(f"Download completed but file not found: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to download {model_name} ({'dia' if dia else 'nodia'}) model: {e}")
            logger.info("Creating dummy model as fallback")
            return self._create_dummy_model(str(model_path), model_name, f"Download failed: {str(e)}")
    
    def _create_dummy_model(self, model_path, model_name, error_msg):
        """Create a dummy model file for testing purposes."""
        logger.info(f"Creating dummy model file: {model_path}")
        dummy_model_state = {
            'dummy': True,
            'model_name': model_name,
            'message': f'Dummy model - {error_msg}'
        }
        torch.save(dummy_model_state, model_path)
        return model_path
    
    def get_model_config(self, model_name):
        """Get model configuration for a specific model."""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        return self.MODEL_CONFIGS[model_name]
    
    def list_available_models(self):
        """List all available model names."""
        return list(self.MODEL_CONFIGS.keys())
