import math
import numpy as np

def linear(t: float) -> float:
    return float(np.clip(t, 0.0, 1.0))


def ease_in_quad(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t


def ease_out_quad(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * (2.0 - t)


def ease_in_out_quad(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return 2.0 * t * t
    return -1.0 + (4.0 - 2.0 * t) * t


def ease_in_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t


def ease_out_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    t -= 1.0
    return t * t * t + 1.0


def ease_in_out_cubic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return 4.0 * t * t * t
    t -= 1.0
    return 4.0 * t * t * t + 1.0


def ease_out_elastic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if t == 0.0 or t == 1.0:
        return t
    p = 0.3
    s = p / 4.0
    return math.pow(2.0, -10.0 * t) * math.sin((t - s) * (2.0 * math.pi) / p) + 1.0


def ease_out_back(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    s = 1.70158
    t -= 1.0
    return t * t * ((s + 1.0) * t + s) + 1.0


def ease_in_out_sine(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - math.cos(t * math.pi))


def ease_out_expo(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    if t == 1.0:
        return 1.0
    return 1.0 - math.pow(2.0, -10.0 * t)
