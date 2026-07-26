import logging
import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple
from config.settings import Settings
from tracking.hand_tracker import HandData
from gesture.detector import GestureDetector, PinchState, HandPose
from gesture.debounce import Debouncer
from animations.animator import AnimationManager, Tween
from animations import easing
from utils.math_utils import map_range, lerp

logger = logging.getLogger(__name__)

class AppState(Enum):
    IDLE = "idle"
    CUBE_APPEARING = "cube_appearing"
    CUBE_ACTIVE = "cube_active"
    CUBE_SHRINKING = "cube_shrinking"
    MODEL_CYCLING = "model_cycling"


@dataclass
class StateContext:
    # Cube scales/opacity
    cube_scale: float = 0.0
    cube_target_scale: float = 1.0
    cube_opacity: float = 0.0
    
    # Model indexes/opacity/scale
    model_index: int = -1
    model_opacity: float = 0.0
    model_scale: float = 0.0
    
    # Track model transitions
    transition_progress: float = 0.0
    total_models: int = 0
    current_model_name: str = ""
    
    # Positions and gestures
    head_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    pinch_distance: float = 0.0
    
    # Continuous animations
    cube_rotation: float = 0.0
    model_rotation: float = 0.0
    manual_rotation_y: float = 0.0
    manual_rotation_x: float = 0.0
    cube_float_offset: float = 0.0


class StateMachine:
    """Finite State Machine (FSM) that governs the interaction states,
    gesture interpretation, and transition animations.
    """
    
    def __init__(self, settings: Settings, animation_manager: AnimationManager):
        self._settings = settings
        self._anim = animation_manager
        
        self._state = AppState.IDLE
        self._context = StateContext()

        # Debouncers for stable state triggers
        self._pinch_open_debouncer = Debouncer(
            required_duration_ms=settings.gesture_debounce_ms
        )
        self._pinch_close_debouncer = Debouncer(
            required_duration_ms=settings.gesture_debounce_ms
        )
        
        # Fist state memory for the fist -> open palm sequence detection
        self._was_fist = False
        self._right_hand_lost_timer = 0.0
        self._fist_open_cooldown_timer = 0.0
        self._elapsed_time = 0.0
        
        # EMA filter factors for smooth tracking
        self._head_smoothing = settings.smoothing_factor

    def rotate_model_manual(self, delta_y: float = 90.0, delta_x: float = 0.0) -> None:
        """Rotates the active 3D model by 90 degree increments manually."""
        self._context.manual_rotation_y = (self._context.manual_rotation_y + delta_y) % 360.0
        self._context.manual_rotation_x = (self._context.manual_rotation_x + delta_x) % 360.0
        logger.info(
            "Model rotated by 90°! Current manual offsets: Y=%.0f°, X=%.0f°",
            self._context.manual_rotation_y, self._context.manual_rotation_x
        )

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def context(self) -> StateContext:
        return self._context

    def update(
        self,
        dt: float,
        left_hand: Optional[HandData],
        right_hand: Optional[HandData],
        head_pos: Optional[np.ndarray],
        gesture_detector: GestureDetector,
        model_count: int,
    ) -> None:
        """Main FSM tick. Updates the current state and handles inputs/transitions."""
        self._elapsed_time += dt
        self._context.total_models = model_count

        # 1. Update Head Tracking (EMA filter)
        if head_pos is not None:
            # Smoothly interpolate head position to minimize frame jitter
            self._context.head_position = (
                self._head_smoothing * head_pos
                + (1.0 - self._head_smoothing) * self._context.head_position
            )

        # 2. Continuous rotation animations
        self._context.cube_rotation += self._settings.cube_rotation_speed * dt
        self._context.model_rotation += self._settings.model_rotation_speed * dt
        
        # Hover float offset (sine wave)
        self._context.cube_float_offset = (
            math.sin(self._elapsed_time * self._settings.cube_float_speed)
            * self._settings.cube_float_amplitude
        )

        # 3. Process Right Hand Gestures (Fist -> Open for model cycling)
        if self._fist_open_cooldown_timer > 0.0:
            self._fist_open_cooldown_timer -= dt
            
        right_hand_open_palm = False
        if right_hand is not None:
            self._right_hand_lost_timer = 0.0
            if self._fist_open_cooldown_timer <= 0.0:
                pose = gesture_detector.detect_hand_pose(right_hand)
                if pose.is_fist:
                    self._was_fist = True
                elif pose.is_open and self._was_fist:
                    # Sequence detected: Fist -> Open palm! Trigger model cycle
                    right_hand_open_palm = True
                    self._was_fist = False
                    self._fist_open_cooldown_timer = self._settings.fist_open_cooldown_ms / 1000.0
        else:
            # Tolerant reset: only clear was_fist memory if tracking is lost for > 1.0 second
            self._right_hand_lost_timer += dt
            if self._right_hand_lost_timer > 1.0:
                self._was_fist = False

        # 4. State-specific Updates
        if self._state == AppState.IDLE:
            # Reset scaling/opacity in idle
            self._context.cube_scale = 0.0
            self._context.cube_opacity = 0.0
            self._context.model_opacity = 0.0
            self._context.model_scale = 0.0
            self._pinch_close_debouncer.reset()

            # Listen for left hand pinch starting the cube creation
            if left_hand is not None:
                pinch = gesture_detector.detect_pinch(left_hand)
                self._context.pinch_distance = pinch.distance
                
                # If they cross pinch threshold (thumb/index apart), trigger appear
                pinch_opened = pinch.distance > self._settings.pinch_open_threshold
                if self._pinch_open_debouncer.update(pinch_opened):
                    self._transition_to(AppState.CUBE_APPEARING)
            else:
                self._pinch_open_debouncer.reset()

        elif self._state == AppState.CUBE_APPEARING:
            # Animation driven scale & opacity
            self._context.cube_scale = self._anim.get_value("cube_scale", 0.0)
            self._context.cube_opacity = self._anim.get_value("cube_opacity", 0.0)
            
            # Keep model off or slightly fading in tandem if first model is selected
            if self._context.model_index >= 0:
                self._context.model_opacity = self._context.cube_opacity
                self._context.model_scale = self._context.cube_scale

        elif self._state == AppState.CUBE_ACTIVE:
            # Size mapping: Left pinch distance maps directly to cube scale
            if left_hand is not None:
                pinch = gesture_detector.detect_pinch(left_hand)
                self._context.pinch_distance = pinch.distance
                
                # Map pinch distance to scale
                target_scale = map_range(
                    pinch.distance,
                    self._settings.pinch_close_threshold,
                    self._settings.pinch_open_threshold * 2.5,
                    self._settings.pinch_min_scale,
                    self._settings.pinch_max_scale
                )
                self._context.cube_scale = lerp(self._context.cube_scale, target_scale, 0.2)
                self._context.cube_opacity = 1.0
                
                # Active model scale/opacity mirrors cube scale
                if self._context.model_index >= 0 and not self._anim.is_active("model_transition"):
                    self._context.model_scale = self._context.cube_scale
                    self._context.model_opacity = 1.0

                # Check if pinch closed below the close threshold
                pinch_closed = pinch.distance < self._settings.pinch_close_threshold
                if self._pinch_close_debouncer.update(pinch_closed):
                    self._transition_to(AppState.CUBE_SHRINKING)
            else:
                # If left hand is lost, slowly float down or decay size, or just return to idle
                # We retain the last size but decay scale slightly or wait for hand re-detection
                self._pinch_close_debouncer.reset()

            # Cycle model if right hand triggers next model sequence
            if right_hand_open_palm and model_count > 0:
                self._cycle_model()

        elif self._state == AppState.MODEL_CYCLING:
            # Lock sizes and manage transition animations
            # Update cube scale from left hand if still available
            if left_hand is not None:
                pinch = gesture_detector.detect_pinch(left_hand)
                self._context.pinch_distance = pinch.distance
                target_scale = map_range(
                    pinch.distance,
                    self._settings.pinch_close_threshold,
                    self._settings.pinch_open_threshold * 2.5,
                    self._settings.pinch_min_scale,
                    self._settings.pinch_max_scale
                )
                self._context.cube_scale = lerp(self._context.cube_scale, target_scale, 0.2)
                
            progress = self._anim.get_value("model_transition", 1.0)
            self._context.transition_progress = progress
            self._context.model_opacity = progress
            self._context.model_scale = self._context.cube_scale * progress

        elif self._state == AppState.CUBE_SHRINKING:
            # Animation driven shrink scale & opacity
            self._context.cube_scale = self._anim.get_value("cube_scale", 1.0)
            self._context.cube_opacity = self._anim.get_value("cube_opacity", 1.0)
            
            # Model decays in sync
            self._context.model_scale = self._context.cube_scale
            self._context.model_opacity = self._context.cube_opacity

    def _cycle_model(self) -> None:
        """Transitions to the next 3D model with scaling/fade transition."""
        if self._context.total_models <= 0:
            return
            
        self._transition_to(AppState.MODEL_CYCLING)
        
        # Transition pattern:
        # 1. Shrink/fade out current model
        # 2. Increment index
        # 3. Grow/fade in new model
        
        def start_fade_in():
            # Target next index
            next_idx = (self._context.model_index + 1) % self._context.total_models
            self._context.model_index = next_idx
            
            self._anim.create_tween(
                "model_transition",
                0.0,
                1.0,
                self._settings.model_transition_duration,
                easing.ease_out_elastic,
                on_complete=lambda: self._transition_to(AppState.CUBE_ACTIVE)
            )
            
        # Shrink current model out first
        self._anim.create_tween(
            "model_transition",
            1.0,
            0.0,
            self._settings.model_transition_duration / 2.0,
            easing.ease_in_quad,
            on_complete=start_fade_in
        )

    def _transition_to(self, new_state: AppState) -> None:
        """Manages state transitions and launches transition animations."""
        logger.info(f"Transitioning FSM state: {self._state.name} -> {new_state.name}")
        old_state = self._state
        self._state = new_state
        
        if new_state == AppState.CUBE_APPEARING:
            self._pinch_open_debouncer.reset()
            # Animate cube scaling up from 0 to 1
            self._anim.create_tween(
                "cube_scale",
                0.0,
                1.0,
                self._settings.cube_appear_duration,
                easing.ease_out_elastic,
                on_complete=lambda: self._transition_to(AppState.CUBE_ACTIVE)
            )
            self._anim.create_tween(
                "cube_opacity",
                0.0,
                1.0,
                self._settings.cube_appear_duration,
                easing.ease_out_quad
            )
            # If no model is active, activate the first one
            if self._context.model_index < 0 and self._context.total_models > 0:
                self._context.model_index = 0

        elif new_state == AppState.CUBE_ACTIVE:
            self._pinch_close_debouncer.reset()
            self._pinch_open_debouncer.reset()
            self._context.cube_opacity = 1.0
            
            # Fully restore active model parameters in case they got interrupted
            if self._context.model_index >= 0:
                self._context.model_opacity = 1.0
                self._context.model_scale = self._context.cube_scale

        elif new_state == AppState.CUBE_SHRINKING:
            self._pinch_close_debouncer.reset()
            # Animate scale down from current size to 0
            self._anim.create_tween(
                "cube_scale",
                self._context.cube_scale,
                0.0,
                self._settings.cube_shrink_duration,
                easing.ease_in_cubic,
                on_complete=lambda: self._transition_to(AppState.IDLE)
            )
            self._anim.create_tween(
                "cube_opacity",
                self._context.cube_opacity,
                0.0,
                self._settings.cube_shrink_duration,
                easing.ease_in_quad
            )

        elif new_state == AppState.IDLE:
            self._context.cube_scale = 0.0
            self._context.cube_opacity = 0.0
            self._context.model_scale = 0.0
            self._context.model_opacity = 0.0
            self._context.model_index = -1  # Reset model index on close

    def reset(self) -> None:
        """Reset state machine context completely."""
        self._state = AppState.IDLE
        self._context = StateContext()
        self._pinch_open_debouncer.reset()
        self._pinch_close_debouncer.reset()
        self._anim.cancel_all()
        self._was_fist = False
        self._fist_open_cooldown_timer = 0.0
        self._elapsed_time = 0.0
