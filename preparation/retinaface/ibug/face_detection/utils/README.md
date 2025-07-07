# Face Detection Utilities

## Purpose

The face detection utilities module provides additional tools for face processing beyond basic detection and landmark extraction. This module includes head pose estimation and simple face tracking capabilities to enhance face detection workflows.

## Folder Structure

```
utils/
├── head_pose_estimator.py            # Head pose estimation from landmarks
├── simple_face_tracker.py            # Simple face tracking across frames
├── data/                             # Utility data files
│   └── bfm_lms.npy                   # BFM-derived landmark reference
└── __init__.py
```

## File Descriptions

### Core Utilities

- **`head_pose_estimator.py`**: `HeadPoseEstimator` class for estimating head pose from facial landmarks
- **`simple_face_tracker.py`**: `SimpleFaceTracker` class for tracking faces across video frames

### Data Files

- **`data/bfm_lms.npy`**: Basel Face Model (BFM) derived landmark reference points for 3D pose estimation

## Internal Usage

### Head Pose Estimation

```python
from preparation.retinaface.ibug.face_detection.utils import HeadPoseEstimator

# Initialize pose estimator
estimator = HeadPoseEstimator()

# Estimate pose from landmarks
pitch, yaw, roll = estimator(landmarks, image_width, image_height)
```

### Face Tracking

```python
from preparation.retinaface.ibug.face_detection.utils import SimpleFaceTracker

# Initialize tracker
tracker = SimpleFaceTracker(iou_threshold=0.4)

# Track faces across frames
face_ids = tracker(face_boxes)
```

## Key Components

### HeadPoseEstimator Class

- **3D Pose Estimation**: Estimates head pose from 2D facial landmarks
- **PnP Solver**: Uses OpenCV's EPnP algorithm for pose estimation
- **Landmark Support**: Works with 68-point, 49-point, and 51-point landmarks
- **Camera Model**: Supports custom camera calibration matrices

### SimpleFaceTracker Class

- **IoU-Based Tracking**: Tracks faces using Intersection over Union (IoU) similarity
- **ID Assignment**: Assigns consistent IDs to faces across frames
- **Minimum Face Size**: Configurable minimum face size threshold
- **Tracklet Management**: Maintains active tracklets for face continuity

## Head Pose Estimation

### Supported Landmark Formats

- **68-Point**: Full facial landmarks (eyes, nose, mouth, jaw)
- **49-Point**: Subset of 68-point landmarks
- **51-Point**: Alternative landmark configuration

### Pose Output

- **Pitch**: Head rotation around X-axis (nodding)
- **Yaw**: Head rotation around Y-axis (shaking)
- **Roll**: Head rotation around Z-axis (tilting)

### Camera Model

- **Default**: Simple camera model based on image dimensions
- **Custom**: Support for custom camera calibration matrices
- **Focal Length**: Estimated from image dimensions if not provided

## Face Tracking

### Tracking Algorithm

- **IoU Similarity**: Measures overlap between face bounding boxes
- **Hungarian Algorithm**: Optimal assignment of detections to tracklets
- **Tracklet Management**: Automatic creation and deletion of tracklets

### Configuration Parameters

- **`iou_threshold`**: Minimum IoU for face association (default: 0.4)
- **`minimum_face_size`**: Minimum face size for tracking (default: 0.0)

### Tracking Output

- **Face IDs**: Consistent integer IDs for tracked faces
- **None Values**: Returned for untracked faces
- **Tracklet State**: Active/inactive tracklet management

## Internal Usage Tips

### Head Pose Estimation

- Use appropriate landmark format for your detection system
- Consider camera calibration for accurate pose estimation
- Validate pose estimates for extreme values
- Apply temporal smoothing for video sequences

### Face Tracking

- Adjust IoU threshold based on detection stability
- Use minimum face size to filter noise
- Reset tracker for scene changes
- Consider temporal consistency in tracking decisions

### Integration Points

- Designed to work with upstream face detection and landmark extraction
- Compatible with video processing pipelines
- Maintains coordinate system consistency
- Supports batch processing for multiple faces

## Performance Characteristics

### Head Pose Estimation

- **Speed**: ~1000+ FPS for landmark-based pose estimation
- **Accuracy**: Good accuracy for frontal and moderate pose variations
- **Robustness**: Handles various lighting conditions

### Face Tracking

- **Speed**: ~10,000+ FPS for IoU-based tracking
- **Memory**: Minimal memory footprint
- **Scalability**: Handles multiple faces efficiently

## Dependencies

- OpenCV
- NumPy
- SciPy (for linear assignment)
- BFM landmark reference data

## Notes for Contributors

- Head pose estimation requires accurate facial landmarks
- Face tracking performance depends on detection quality
- The module supports both single frame and video sequence processing
- Coordinate systems are maintained consistently across components
- Consider temporal smoothing for video applications
