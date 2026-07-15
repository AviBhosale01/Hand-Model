import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Tuple, Any, Dict

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Settings:
    # Camera settings
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    # Window settings
    window_width: int = 1280
    window_height: int = 720
    window_title: str = "AR Holographic Viewer"
    window_vsync: bool = True
    window_fullscreen: bool = False

    # Gesture settings
    pinch_open_threshold: float = 0.08
    pinch_close_threshold: float = 0.03
    pinch_min_scale: float = 0.3
    pinch_max_scale: float = 2.0
    gesture_debounce_ms: int = 200
    fist_open_cooldown_ms: int = 500

    # Tracking settings
    hand_detection_confidence: float = 0.7
    hand_tracking_confidence: float = 0.6
    face_detection_confidence: float = 0.6
    smoothing_factor: float = 0.3
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007

    # Cube settings
    cube_base_size: float = 0.4
    cube_rotation_speed: float = 15.0
    cube_float_amplitude: float = 0.02
    cube_float_speed: float = 1.5
    cube_appear_duration: float = 0.5
    cube_shrink_duration: float = 0.3

    # Model settings
    model_directory: str = "assets"
    model_rotation_speed: float = 30.0
    model_transition_duration: float = 0.5

    # Visual settings
    glow_color: Tuple[float, float, float] = (0.0, 0.8, 1.0)
    glow_intensity: float = 1.5
    bloom_threshold: float = 0.6
    bloom_intensity: float = 0.8
    bloom_blur_passes: int = 5
    particle_count: int = 200

    # Debug settings
    debug_enabled: bool = False
    show_fps: bool = True
    show_landmarks: bool = False
    log_level: str = "INFO"
    log_file: str = "logs/app.log"


def load_settings(path: str = "config.yaml") -> Settings:
    """Loads configuration from yaml file and returns Settings instance.
    Falls back to defaults if the file is missing or invalid.
    """
    defaults = {}
    if not os.path.exists(path):
        logger.warning(f"Config file {path} not found. Using defaults.")
        return Settings()

    try:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        
        # Flatten nested sections
        flat_dict: Dict[str, Any] = {}
        
        # Camera
        camera = config_dict.get("camera", {})
        flat_dict["camera_index"] = camera.get("index", 0)
        flat_dict["camera_width"] = camera.get("width", 1280)
        flat_dict["camera_height"] = camera.get("height", 720)
        flat_dict["camera_fps"] = camera.get("fps", 30)
        
        # Window
        window = config_dict.get("window", {})
        flat_dict["window_width"] = window.get("width", 1280)
        flat_dict["window_height"] = window.get("height", 720)
        flat_dict["window_title"] = window.get("title", "AR Holographic Viewer")
        flat_dict["window_vsync"] = window.get("vsync", True)
        flat_dict["window_fullscreen"] = window.get("fullscreen", False)
        
        # Gesture
        gesture = config_dict.get("gesture", {})
        flat_dict["pinch_open_threshold"] = gesture.get("pinch_open_threshold", 0.08)
        flat_dict["pinch_close_threshold"] = gesture.get("pinch_close_threshold", 0.03)
        flat_dict["pinch_min_scale"] = gesture.get("pinch_min_scale", 0.3)
        flat_dict["pinch_max_scale"] = gesture.get("pinch_max_scale", 2.0)
        flat_dict["gesture_debounce_ms"] = gesture.get("debounce_ms", 200)
        flat_dict["fist_open_cooldown_ms"] = gesture.get("fist_open_cooldown_ms", 500)
        
        # Tracking
        tracking = config_dict.get("tracking", {})
        flat_dict["hand_detection_confidence"] = tracking.get("hand_detection_confidence", 0.7)
        flat_dict["hand_tracking_confidence"] = tracking.get("hand_tracking_confidence", 0.6)
        flat_dict["face_detection_confidence"] = tracking.get("face_detection_confidence", 0.6)
        flat_dict["smoothing_factor"] = tracking.get("smoothing_factor", 0.3)
        flat_dict["one_euro_min_cutoff"] = tracking.get("one_euro_min_cutoff", 1.0)
        flat_dict["one_euro_beta"] = tracking.get("one_euro_beta", 0.007)
        
        # Cube
        cube = config_dict.get("cube", {})
        flat_dict["cube_base_size"] = cube.get("base_size", 0.4)
        flat_dict["cube_rotation_speed"] = cube.get("rotation_speed", 15.0)
        flat_dict["cube_float_amplitude"] = cube.get("float_amplitude", 0.02)
        flat_dict["cube_float_speed"] = cube.get("float_speed", 1.5)
        flat_dict["cube_appear_duration"] = cube.get("appear_duration", 0.5)
        flat_dict["cube_shrink_duration"] = cube.get("shrink_duration", 0.3)
        
        # Model
        model = config_dict.get("model", {})
        flat_dict["model_directory"] = model.get("directory", "assets")
        flat_dict["model_rotation_speed"] = model.get("rotation_speed", 30.0)
        flat_dict["model_transition_duration"] = model.get("transition_duration", 0.5)
        
        # Visual
        visual = config_dict.get("visual", {})
        glow_color = visual.get("glow_color", [0.0, 0.8, 1.0])
        flat_dict["glow_color"] = tuple(glow_color) if len(glow_color) == 3 else (0.0, 0.8, 1.0)
        flat_dict["glow_intensity"] = visual.get("glow_intensity", 1.5)
        flat_dict["bloom_threshold"] = visual.get("bloom_threshold", 0.6)
        flat_dict["bloom_intensity"] = visual.get("bloom_intensity", 0.8)
        flat_dict["bloom_blur_passes"] = visual.get("bloom_blur_passes", 5)
        flat_dict["particle_count"] = visual.get("particle_count", 200)
        
        # Debug
        debug = config_dict.get("debug", {})
        flat_dict["debug_enabled"] = debug.get("enabled", False)
        flat_dict["show_fps"] = debug.get("show_fps", True)
        flat_dict["show_landmarks"] = debug.get("show_landmarks", False)
        flat_dict["log_level"] = debug.get("log_level", "INFO")
        flat_dict["log_file"] = debug.get("log_file", "logs/app.log")
        
        return Settings(**flat_dict)
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}. Using defaults.")
        return Settings()
