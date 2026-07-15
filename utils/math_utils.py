import numpy as np
from typing import Tuple

def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolates between a and b by t."""
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamps a value between a minimum and maximum range."""
    return max(min_val, min(value, max_val))


def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Linearly maps a value from one range [in_min, in_max] to another [out_min, out_max]."""
    if abs(in_max - in_min) < 1e-6:
        return out_min
    normalized = (value - in_min) / (in_max - in_min)
    return out_min + normalized * (out_max - out_min)


def normalize_mesh_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fits mesh vertices perfectly inside a unit cube [-0.5, 0.5]^3 centered at origin.
    Returns:
        Tuple[normalized_vertices, scale_factor, center_offset]
    """
    if vertices.size == 0:
        return vertices, 1.0, np.zeros(3)

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2.0
    
    # Translate to origin
    translated = vertices - center
    
    # Compute bounding box dimensions
    dims = max_coords - min_coords
    max_dim = np.max(dims)
    
    if max_dim < 1e-6:
        scale = 1.0
    else:
        # Scale to fit unit cube (length of 1.0 along the largest axis)
        scale = 1.0 / max_dim
        
    normalized = translated * scale
    return normalized.astype(np.float32), scale, center.astype(np.float32)


def screen_to_ndc(x: float, y: float, width: int, height: int) -> Tuple[float, float]:
    """Converts pixel coordinate (top-left origin) to Normalized Device Coordinates (NDC, [-1, 1])."""
    ndc_x = (x / width) * 2.0 - 1.0
    ndc_y = 1.0 - (y / height) * 2.0  # invert Y coordinate
    return ndc_x, ndc_y


def smooth_damp(
    current: float,
    target: float,
    current_velocity: float,
    smooth_time: float,
    dt: float,
    max_speed: float = float("inf"),
) -> Tuple[float, float]:
    """Gradually changes a value towards a desired goal over time.
    Calculated using a spring-damper model similar to Unity's SmoothDamp.
    Returns:
        Tuple[new_value, new_velocity]
    """
    smooth_time = max(0.0001, smooth_time)
    num = 2.0 / smooth_time
    num2 = num * dt
    num3 = 1.0 / (1.0 + num2 + 0.48 * num2 * num2 + 0.235 * num2 * num2 * num2)
    
    num4 = current - target
    num5 = target
    
    max_change = max_speed * smooth_time
    num4 = clamp(num4, -max_change, max_change)
    target = current - num4
    
    num6 = (current_velocity + num * num4) * dt
    current_velocity = (current_velocity - num * num6) * num3
    
    new_value = target + (num4 + num6) * num3
    
    if (num5 - current > 0.0) == (new_value > num5):
        new_value = num5
        current_velocity = 0.0
        
    return new_value, current_velocity
