import logging
from typing import Callable, Dict, Optional, List

logger = logging.getLogger(__name__)

class Tween:
    """Calculates interpolated values between start and end over a specified duration."""
    
    def __init__(
        self,
        start: float,
        end: float,
        duration: float,
        easing: Callable[[float], float],
        on_complete: Optional[Callable[[], None]] = None,
    ):
        self._start = start
        self._end = end
        self._duration = max(0.0001, duration)
        self._easing = easing
        self._on_complete = on_complete
        
        self._elapsed = 0.0
        self._value = start
        self._is_complete = False

    def update(self, dt: float) -> float:
        """Update elapsed time and compute the current value."""
        if self._is_complete:
            return self._value

        self._elapsed += dt
        t = self._elapsed / self._duration
        
        if t >= 1.0:
            t = 1.0
            self._value = self._end
            self._is_complete = True
            if self._on_complete:
                try:
                    self._on_complete()
                except Exception as e:
                    logger.error(f"Error in Tween completion callback: {e}", exc_info=True)
        else:
            e = self._easing(t)
            self._value = self._start + (self._end - self._start) * e
            
        return self._value

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @property
    def value(self) -> float:
        return self._value


class AnimationManager:
    """Manages collection of active tweens."""
    
    def __init__(self):
        self._active_tweens: Dict[str, Tween] = {}

    def create_tween(
        self,
        name: str,
        start: float,
        end: float,
        duration: float,
        easing: Callable[[float], float],
        on_complete: Optional[Callable[[], None]] = None,
    ) -> Tween:
        """Create a new tween or overwrite an existing one with the same name."""
        tween = Tween(start, end, duration, easing, on_complete)
        self._active_tweens[name] = tween
        return tween

    def update(self, dt: float) -> None:
        """Update all active tweens, remove completed ones."""
        completed_keys = []
        for name, tween in list(self._active_tweens.items()):
            tween.update(dt)
            if tween.is_complete:
                completed_keys.append(name)
                
        for name in completed_keys:
            if name in self._active_tweens:
                del self._active_tweens[name]

    def get_value(self, name: str, default: float = 0.0) -> float:
        """Get the current value of a running tween. Returns default if not found."""
        if name in self._active_tweens:
            return self._active_tweens[name].value
        return default

    def is_active(self, name: str) -> bool:
        """Checks if a tween with the given name is currently running."""
        return name in self._active_tweens

    def cancel(self, name: str) -> None:
        """Cancel a running tween by name."""
        if name in self._active_tweens:
            del self._active_tweens[name]

    def cancel_all(self) -> None:
        """Cancel all running tweens."""
        self._active_tweens.clear()
