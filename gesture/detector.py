import numpy as np
import logging
from dataclasses import dataclass
from config.settings import Settings
from tracking.hand_tracker import HandData

logger = logging.getLogger(__name__)

@dataclass
class PinchState:
    is_pinching: bool
    distance: float  # Normalized Euclidean distance
    raw_distance: float

@dataclass
class HandPose:
    is_fist: bool
    is_open: bool
    confidence: float

class GestureDetector:
    """Processes hand landmarks to detect specific gestures (pinch, fist, open palm)."""
    
    def __init__(self, settings: Settings):
        self._pinch_open_threshold = settings.pinch_open_threshold
        self._pinch_close_threshold = settings.pinch_close_threshold

    def detect_pinch(self, hand: HandData) -> PinchState:
        """Measures distance between thumb tip (4) and index tip (8)
        normalized by the hand bounding box diagonal for scale invariance.
        """
        landmarks = hand.landmarks
        
        # Extents to calculate normalized diagonal
        min_coords = np.min(landmarks, axis=0)
        max_coords = np.max(landmarks, axis=0)
        diagonal = np.linalg.norm(max_coords - min_coords)
        
        if diagonal < 1e-5:
            diagonal = 1.0

        # Thumb tip (4) and Index tip (8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        raw_dist = np.linalg.norm(thumb_tip - index_tip)
        norm_dist = raw_dist / diagonal
        
        is_pinching = norm_dist < self._pinch_open_threshold
        
        return PinchState(
            is_pinching=is_pinching,
            distance=norm_dist,
            raw_distance=raw_dist
        )

    def detect_hand_pose(self, hand: HandData) -> HandPose:
        """Classifies if a hand is in a closed fist or an open palm.
        Fist: Finger tips (8, 12, 16, 20) are closer to wrist (0) or MCPs than PIPs.
        Open Palm: Fingers are extended.
        """
        landmarks = hand.landmarks
        wrist = landmarks[0]
        
        # Landmark indices:
        # Index: Tip (8), PIP (6), MCP (5)
        # Middle: Tip (12), PIP (10), MCP (9)
        # Ring: Tip (16), PIP (14), MCP (13)
        # Pinky: Tip (20), PIP (18), MCP (17)
        fingers = [
            (8, 6, 5),   # Index
            (12, 10, 9), # Middle
            (16, 14, 13),# Ring
            (20, 18, 17) # Pinky
        ]
        
        curled_count = 0
        extended_count = 0
        
        for tip_idx, pip_idx, mcp_idx in fingers:
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            
            # Rotation-invariant distance-based check:
            # A finger is curled if the tip is closer to the wrist or the base MCP joint than the PIP joint is.
            dist_tip_wrist = np.linalg.norm(tip - wrist)
            dist_pip_wrist = np.linalg.norm(pip - wrist)
            dist_tip_mcp = np.linalg.norm(tip - mcp)
            dist_pip_mcp = np.linalg.norm(pip - mcp)
            
            if dist_tip_wrist < dist_pip_wrist or dist_tip_mcp < dist_pip_mcp:
                curled_count += 1
            else:
                extended_count += 1
                
        # Thumb check (Thumb tip 4 vs Index MCP 5 / Pinky MCP 17)
        # Just simple threshold on thumb distance to middle finger base (9)
        thumb_tip = landmarks[4]
        middle_mcp = landmarks[9]
        dist_thumb_middle = np.linalg.norm(thumb_tip - middle_mcp)
        
        # Calculate bounding diagonal for normalization
        min_coords = np.min(landmarks, axis=0)
        max_coords = np.max(landmarks, axis=0)
        diagonal = np.linalg.norm(max_coords - min_coords)
        if diagonal < 1e-5:
            diagonal = 1.0
            
        thumb_curled = (dist_thumb_middle / diagonal) < 0.25
        if thumb_curled:
            curled_count += 1
        else:
            extended_count += 1
            
        # Classify based on majority vote
        is_fist = curled_count >= 4
        is_open = extended_count >= 4
        
        confidence = max(curled_count, extended_count) / 5.0
        
        return HandPose(
            is_fist=is_fist,
            is_open=is_open,
            confidence=confidence
        )
