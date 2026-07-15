import cv2
import queue
import logging
import threading
import numpy as np
from typing import Optional, Tuple
from config.settings import Settings

logger = logging.getLogger(__name__)

class CameraCapture:
    """Threaded camera frame grabber. Runs a background loop to pull frames
    continuously to minimize frame latency and avoid blocking the render loop.
    """
    
    def __init__(self, settings: Settings):
        self._camera_index = settings.camera_index
        self._target_width = settings.camera_width
        self._target_height = settings.camera_height
        self._target_fps = settings.camera_fps
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._latest_frame: Optional[np.ndarray] = None
        self._actual_size: Tuple[int, int] = (0, 0)

    def start(self) -> bool:
        """Opens the camera device and starts the reader thread. Returns True if successful."""
        logger.info(f"Opening camera index {self._camera_index}...")
        
        # Set DirectShow backend on Windows if appropriate for faster initialization
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW if cv2.os.name == 'nt' else cv2.CAP_ANY)
        
        if not self._cap.isOpened():
            # Retry with default backend
            self._cap = cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera index {self._camera_index}")
                return False
                
        # Attempt configuration
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_size = (width, height)
        
        logger.info(f"Camera opened. Resolution configured: {width}x{height} @ {self._cap.get(cv2.CAP_PROP_FPS)} FPS")
        
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="CameraCaptureThread")
        self._thread.start()
        
        return True

    def _reader_loop(self) -> None:
        """Background thread main loop for grabbing frames."""
        while self._running:
            if self._cap is None:
                break
                
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue
                
            with self._lock:
                # Store frame in OpenCV default BGR format
                self._latest_frame = frame

    def get_frame(self) -> Optional[np.ndarray]:
        """Retrieves the latest BGR frame. Thread-safe."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """Retrieves the latest frame converted to RGB. Thread-safe."""
        bgr = self.get_frame()
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        """Stops the reader thread and releases the camera device."""
        logger.info("Releasing camera capture...")
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
            
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._latest_frame = None
            
        logger.info("Camera capture released.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._actual_size
