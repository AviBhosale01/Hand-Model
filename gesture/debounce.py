import time

class Debouncer:
    """Confirms a condition remains True for a specified duration (in milliseconds)
    before triggering. Includes an optional cooldown period.
    """
    
    def __init__(self, required_duration_ms: int = 200, cooldown_ms: int = 0):
        self._required_duration = required_duration_ms / 1000.0
        self._cooldown_duration = cooldown_ms / 1000.0
        
        self._held_start: float = 0.0
        self._last_trigger_time: float = 0.0
        
        self._is_active = False
        self._triggered = False

    def update(self, condition: bool, current_time: float = None) -> bool:
        """Evaluates condition. Returns True on the exact frame it triggers."""
        now = time.perf_counter() if current_time is None else current_time
        
        # Check cooldown
        if now - self._last_trigger_time < self._cooldown_duration:
            self.reset()
            return False

        if condition:
            if not self._is_active:
                self._held_start = now
                self._is_active = True
                self._triggered = False
            
            # Check duration
            elif not self._triggered and (now - self._held_start) >= self._required_duration:
                self._triggered = True
                self._last_trigger_time = now
                return True
        else:
            self.reset()
            
        return False

    def reset(self) -> None:
        """Resets the debouncer internal timer state."""
        self._is_active = False
        self._triggered = False
        self._held_start = 0.0

    @property
    def is_active(self) -> bool:
        """True if the condition is currently being held/evaluated."""
        return self._is_active

    @property
    def progress(self) -> float:
        """How close the active condition is to triggering (0.0 to 1.0)."""
        if not self._is_active or self._triggered:
            return 1.0 if self._triggered else 0.0
        now = time.perf_counter()
        elapsed = now - self._held_start
        return min(1.0, elapsed / self._required_duration)
