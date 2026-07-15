import math
import numpy as np

class OneEuroFilter:
    """Adaptive low-pass filter to smooth noisy values without introducing latency."""
    
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        self._x_prev: float = 0.0
        self._dx_prev: float = 0.0
        self._initialized = False

    def __call__(self, x: float, dt: float) -> float:
        if not self._initialized:
            self._x_prev = x
            self._dx_prev = 0.0
            self._initialized = True
            return x

        # Calculate derivative of signal
        dx = (x - self._x_prev) / dt if dt > 0.0 else 0.0
        
        # Smooth derivative
        alpha_d = self._alpha(dt, self.d_cutoff)
        dx_hat = self._lerp(self._dx_prev, dx, alpha_d)
        
        # Smooth signal based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha_x = self._alpha(dt, cutoff)
        x_hat = self._lerp(self._x_prev, x, alpha_x)
        
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        
        return x_hat

    def reset(self) -> None:
        self._initialized = False

    def _alpha(self, dt: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return dt / (dt + tau) if (dt + tau) > 0.0 else 1.0

    def _lerp(self, a: float, b: float, t: float) -> float:
        return a + (b - a) * t


class VectorSmoother:
    """Applies a OneEuroFilter independently to each dimension of a vector (e.g. 3D positions)."""
    
    def __init__(self, dimensions: int = 3, min_cutoff: float = 1.0, beta: float = 0.007):
        self.filters = [OneEuroFilter(min_cutoff, beta) for _ in range(dimensions)]

    def __call__(self, value: np.ndarray, dt: float) -> np.ndarray:
        result = np.zeros_like(value)
        for i, val in enumerate(value):
            if i < len(self.filters):
                result[i] = self.filters[i](float(val), dt)
        return result

    def reset(self) -> None:
        for f in self.filters:
            f.reset()


class EMAFilter:
    """Simple Exponential Moving Average filter."""
    
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._value = 0.0
        self._initialized = False

    def __call__(self, value: float) -> float:
        if not self._initialized:
            self._value = value
            self._initialized = True
            return value
            
        self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._initialized = False
