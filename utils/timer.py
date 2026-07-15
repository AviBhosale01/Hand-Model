import time
from collections import deque

class FrameTimer:
    """Utility class to measure delta time, rolling average FPS, and elapsed runtime."""
    
    def __init__(self, fps_sample_count: int = 60):
        self._fps_sample_count = fps_sample_count
        self._start_time = time.perf_counter()
        self._last_time = self._start_time
        self._dt = 0.0
        self._frame_count = 0
        self._deltas = deque(maxlen=fps_sample_count)

    def tick(self) -> float:
        """Advance the timer by one frame. Returns the delta time in seconds."""
        current = time.perf_counter()
        self._dt = current - self._last_time
        self._last_time = current
        
        # Guard against zero division/extremely small deltas
        if self._dt <= 0.0:
            self._dt = 1.0 / 60.0
            
        self._deltas.append(self._dt)
        self._frame_count += 1
        return self._dt

    @property
    def dt(self) -> float:
        """Time elapsed since the last tick (in seconds)."""
        return self._dt

    @property
    def fps(self) -> float:
        """Rolling average frames per second."""
        if not self._deltas:
            return 0.0
        avg_dt = sum(self._deltas) / len(self._deltas)
        return 1.0 / avg_dt if avg_dt > 0.0 else 0.0

    @property
    def frame_count(self) -> int:
        """Total number of frames (ticks) registered."""
        return self._frame_count

    @property
    def elapsed(self) -> float:
        """Total time elapsed since the timer was created (in seconds)."""
        return time.perf_counter() - self._start_time
