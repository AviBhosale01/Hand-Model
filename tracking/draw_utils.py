import cv2
import numpy as np
from typing import List, Optional
from tracking.hand_tracker import HandData
from tracking.face_tracker import FaceData

# Standard MediaPipe Hand Connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm MCP connections
    (5, 9), (9, 13), (13, 17)
]

def draw_landmarks(frame_rgb: np.ndarray, hands: List[HandData], face: Optional[FaceData]) -> None:
    """Draws hand skeletons and face mesh points directly onto the camera frame.
    Note: frame_rgb is in RGB color space, so colors should be specified in RGB order.
    """
    h, w, _ = frame_rgb.shape

    # 1. Draw Hand Skeletons
    for hand in hands:
        # Yellow for Left hand, Green for Right hand
        # RGB Color values: Left = Yellow (255, 255, 0), Right = Green (0, 255, 0)
        color = (0, 255, 0) if hand.handedness == "Right" else (255, 255, 0)
        
        # Scale landmarks to pixel coordinates
        pts = []
        for lm in hand.landmarks:
            px = int(lm[0] * w)
            py = int(lm[1] * h)
            pts.append((px, py))
            
        # Draw skeleton connection lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            if start_idx < len(pts) and end_idx < len(pts):
                cv2.line(frame_rgb, pts[start_idx], pts[end_idx], color, 2, cv2.LINE_AA)
                
        # Draw joint dots
        for pt in pts:
            cv2.circle(frame_rgb, pt, 5, color, -1, cv2.LINE_AA)

    # 2. Draw Face Mesh Points
    if face is not None:
        # Draw all 478 cyan face mesh dots (RGB: (0, 255, 255))
        for lm in face.landmarks:
            px = int(lm[0] * w)
            py = int(lm[1] * h)
            cv2.circle(frame_rgb, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)
            
        # Draw the head center position circle (Red, RGB: (255, 0, 0))
        # Landmark 4 in MediaPipe face mesh is the nose tip
        if len(face.landmarks) > 4:
            nose_tip = face.landmarks[4]
            px = int(nose_tip[0] * w)
            py = int(nose_tip[1] * h)
            
            # Red circle (outline)
            cv2.circle(frame_rgb, (px, py), 10, (255, 0, 0), 2, cv2.LINE_AA)
            # Label
            cv2.putText(
                frame_rgb, 
                "Head Pos", 
                (px + 15, py + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 0, 0), 
                1, 
                cv2.LINE_AA
            )
