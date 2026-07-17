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
    landmarks: np.ndarray  # (478, 3) normalized face landmarks
    confidence: float

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_PATH = "assets/face_landmarker.task"

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
    """Wrapper around MediaPipe Tasks Face Landmarker. Uses 478 face landmarks
    to calculate precise bounding boxes, centers, and estimated depth.
    Works out-of-the-box on Python 3.12/3.13.
    """
    
    def __init__(self, settings: Settings):
        _ensure_model_exists(FACE_MODEL_URL, FACE_MODEL_PATH)
        
        base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=settings.face_detection_confidence,
            min_face_presence_confidence=settings.face_detection_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        
        # Reference height of a face in bounding box terms at a standard distance
        self._ref_face_height = 0.28

    def process(self, frame_rgb: np.ndarray) -> Optional[FaceData]:
        """Processes an RGB frame and returns FaceData containing mesh landmarks and center coordinates."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None
            
        # Select first detected face landmarks
        face_landmarks = result.face_landmarks[0]
        
        # Convert landmarks to numpy array (N, 3)
        landmarks = np.zeros((len(face_landmarks), 3), dtype=np.float32)
        for i, lm in enumerate(face_landmarks):
            landmarks[i] = [lm.x, lm.y, lm.z]
            
        # Compute bounding box from landmarks range
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        width = xmax - xmin
        height = ymax - ymin
        
        # Calculate center coordinate of the face bounding box
        center_x = xmin + width / 2.0
        center_y = ymin + height / 2.0
        
        # Depth estimate (larger face = closer, smaller = further)
        depth_est = self._ref_face_height / max(0.01, height)
        depth_est = np.clip(depth_est, 0.5, 4.0)
        
        center = np.array([center_x, center_y, depth_est], dtype=np.float32)
        
        # Confidence score category value (default to 1.0 since landmarks are detected)
        score = 1.0
        
        return FaceData(
            center=center,
            bbox=(xmin, ymin, width, height),
            landmarks=landmarks,
            confidence=score
        )

    def release(self) -> None:
        """Close FaceLandmarker instance."""
        try:
            self._detector.close()
        except Exception as e:
            logger.error(f"Error closing FaceTracker: {e}")
