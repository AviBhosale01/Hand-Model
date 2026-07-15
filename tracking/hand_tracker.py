import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from dataclasses import dataclass
from typing import List, Tuple
from config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class HandData:
    landmarks: np.ndarray  # Shape: (21, 3) normalized x, y, z coordinates
    handedness: str        # "Left" or "Right"
    confidence: float

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = "assets/hand_landmarker.task"

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

class HandTracker:
    """Wrapper around MediaPipe Tasks Hand Landmarker to extract normalized 3D landmarks
    and handedness classification. Works out-of-the-box on Python 3.12/3.13.
    """
    
    def __init__(self, settings: Settings):
        _ensure_model_exists(HAND_MODEL_URL, HAND_MODEL_PATH)
        
        base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=settings.hand_detection_confidence,
            min_hand_presence_confidence=settings.hand_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_rgb: np.ndarray) -> List[HandData]:
        """Processes an RGB frame and returns tracked HandData for up to 2 hands."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        
        tracked_hands: List[HandData] = []
        if not result.hand_landmarks:
            return tracked_hands

        for hand_landmarks, handedness_category in zip(result.hand_landmarks, result.handedness):
            # Parse landmarks into numpy array (21, 3)
            landmarks = np.zeros((21, 3), dtype=np.float32)
            for i, lm in enumerate(hand_landmarks):
                landmarks[i] = [lm.x, lm.y, lm.z]
                
            label = handedness_category[0].category_name  # "Left" or "Right"
            score = handedness_category[0].score
            
            tracked_hands.append(HandData(
                landmarks=landmarks,
                handedness=label,
                confidence=score
            ))
            
        return tracked_hands

    def release(self) -> None:
        """Close MediaPipe HandLandmarker instance."""
        try:
            self._detector.close()
        except Exception as e:
            logger.error(f"Error closing HandTracker: {e}")
