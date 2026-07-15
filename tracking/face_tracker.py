import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class FaceData:
    center: np.ndarray  # (3,) -> x, y in normalized space, z is depth estimate
    bbox: Tuple[float, float, float, float]  # (xmin, ymin, width, height) in normalized space
    confidence: float

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
FACE_MODEL_PATH = "assets/blaze_face_short_range.tflite"

def _ensure_model_exists(url: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        return
    logger.info(f"Downloading MediaPipe model asset from {url} to {dest_path}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        out_file.write(response.read())
    logger.info(f"Successfully downloaded {dest_path}")

class FaceTracker:
    """Wrapper around MediaPipe Tasks Face Detector. Uses the bounding box
    dimensions to estimate the approximate distance (depth) of the user's face.
    Works out-of-the-box on Python 3.12/3.13.
    """
    
    def __init__(self, settings: Settings):
        _ensure_model_exists(FACE_MODEL_URL, FACE_MODEL_PATH)
        
        base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=settings.face_detection_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        
        # Reference height of a face in bounding box terms at a standard distance
        # Used for heuristic depth estimation (inverse relationship)
        self._ref_face_height = 0.25

    def process(self, frame_rgb: np.ndarray) -> Optional[FaceData]:
        """Processes an RGB frame and returns FaceData for the primary face detected."""
        h, w, _ = frame_rgb.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        
        if not result.detections:
            return None
            
        # Select detection with highest score or first detection
        primary_detection = result.detections[0]
        score = primary_detection.categories[0].score if primary_detection.categories else 0.0
        
        # Bounding box is in pixel coordinates
        pixel_bbox = primary_detection.bounding_box
        xmin = pixel_bbox.origin_x / w
        ymin = pixel_bbox.origin_y / h
        width = pixel_bbox.width / w
        height = pixel_bbox.height / h
        
        # Calculate centers
        center_x = xmin + width / 2.0
        center_y = ymin + height / 2.0
        
        # Depth heuristic: larger face = closer, smaller = further
        # When height is self._ref_face_height, depth is 1.0 (arbitrary unit)
        depth_est = self._ref_face_height / max(0.01, height)
        
        # Clip/smooth depth range to prevent anomalies
        depth_est = np.clip(depth_est, 0.5, 4.0)
        
        center = np.array([center_x, center_y, depth_est], dtype=np.float32)
        
        return FaceData(
            center=center,
            bbox=(xmin, ymin, width, height),
            confidence=score
        )

    def release(self) -> None:
        """Close FaceDetector instance."""
        try:
            self._detector.close()
        except Exception as e:
            logger.error(f"Error closing FaceTracker: {e}")
