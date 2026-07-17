"""
Master scene renderer for the AR Holographic Gesture Controlled 3D Object Viewer.

Orchestrates every rendering subsystem — background camera feed, holographic
wireframe cube, 3D model display, particle effects, bloom post-processing,
and HUD text overlay — into a unified per-frame render pipeline.

Rendering order per frame:
    1. Update background texture with latest camera frame.
    2. Clear the screen.
    3. Draw the camera feed as a fullscreen background.
    4. If state ≠ IDLE, capture 3D scene into the bloom FBO:
        a. Wireframe holographic cube.
        b. Loaded 3D model (if visible).
        c. Orbiting energy-spark particles.
    5. Apply bloom post-processing (extract → blur → combine).
    6. Draw HUD overlays (FPS, state, model name).
"""

import logging
import math
from typing import Optional

import numpy as np
import pyrr
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA16F,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glClearColor,
    glEnable,
    glViewport,
)

from config.settings import Settings
from effects.bloom import BloomEffect
from gesture.state_machine import AppState, StateContext
from graphics.gl_utils import (
    create_fbo_with_texture,
    create_fullscreen_quad_vao,
    create_texture_2d,
    create_vao,
    create_vbo,
    create_ebo,
    update_texture_2d,
)
from graphics.shader import ShaderProgram
from models.loader import MeshData
from renderer.background import BackgroundRenderer
from renderer.cube_renderer import CubeRenderer
from renderer.hud_renderer import HUDRenderer
from renderer.model_renderer import ModelRenderer
from renderer.particle_renderer import ParticleRenderer

logger = logging.getLogger(__name__)


class SceneRenderer:
    """Master orchestrator that drives every rendering subsystem each frame.

    The renderer lazily initialises all GPU resources via :meth:`initialize` so
    it can be constructed before an OpenGL context exists.

    Attributes:
        _settings: Application settings.
        _debug_visible: Whether to draw HUD debug overlays.
        _background_shader / _hologram_shader / …: Shader programs.
        _background / _cube / _model / _particles / _hud: Sub-renderers.
        _bloom: Bloom post-processing effect.
        _bloom_extract_shader / _bloom_blur_shader / _bloom_combine_shader:
            Shaders consumed by the bloom effect.
    """

    def __init__(self, settings: Settings) -> None:
        """Create the scene renderer (GPU resources are NOT allocated here).

        Args:
            settings: Application-wide settings dataclass.
        """
        self._settings = settings
        self._debug_visible: bool = settings.debug_enabled
        self._width: int = settings.window_width
        self._height: int = settings.window_height

        # ── Sub-renderers (initialised in ``initialize``) ─────────────────
        self._background: Optional[BackgroundRenderer] = None
        self._cube: Optional[CubeRenderer] = None
        self._model: Optional[ModelRenderer] = None
        self._particles: Optional[ParticleRenderer] = None
        self._hud: Optional[HUDRenderer] = None
        self._bloom: Optional[BloomEffect] = None

        # ── Shaders ───────────────────────────────────────────────────────
        self._background_shader: Optional[ShaderProgram] = None
        self._hologram_shader: Optional[ShaderProgram] = None
        self._model_shader: Optional[ShaderProgram] = None
        self._particle_shader: Optional[ShaderProgram] = None
        self._bloom_extract_shader: Optional[ShaderProgram] = None
        self._bloom_blur_shader: Optional[ShaderProgram] = None
        self._bloom_combine_shader: Optional[ShaderProgram] = None

        logger.info("SceneRenderer constructed (uninitialised)")

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Load shaders, create all sub-renderers, and set GL defaults.

        Must be called once after a valid OpenGL context is available.
        """
        s = self._settings
        w, h = self._width, self._height

        # ── Load shaders ──────────────────────────────────────────────────
        logger.info("Loading shaders …")
        try:
            self._background_shader = ShaderProgram(
                "shaders/background.vert", "shaders/background.frag"
            )
            self._hologram_shader = ShaderProgram(
                "shaders/hologram.vert", "shaders/hologram.frag"
            )
            self._model_shader = ShaderProgram(
                "shaders/model.vert", "shaders/model.frag"
            )
            self._particle_shader = ShaderProgram(
                "shaders/particle.vert", "shaders/particle.frag"
            )
            self._bloom_extract_shader = ShaderProgram(
                "shaders/passthrough.vert", "shaders/bloom_extract.frag"
            )
            self._bloom_blur_shader = ShaderProgram(
                "shaders/passthrough.vert", "shaders/bloom_blur.frag"
            )
            self._bloom_combine_shader = ShaderProgram(
                "shaders/passthrough.vert", "shaders/bloom_combine.frag"
            )
        except Exception:
            logger.exception("Shader compilation failed")
            raise

        # ── Create sub-renderers ──────────────────────────────────────────
        logger.info("Creating sub-renderers …")
        try:
            self._background = BackgroundRenderer(self._background_shader, w, h)
            self._cube = CubeRenderer(self._hologram_shader, s)
            self._model = ModelRenderer(self._model_shader, s)
            self._particles = ParticleRenderer(self._particle_shader, s)
            self._hud = HUDRenderer(w, h)
            self._bloom = BloomEffect(w, h, s)
        except Exception:
            logger.exception("Sub-renderer initialisation failed")
            raise

        # ── Global OpenGL state ───────────────────────────────────────────
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        logger.info("SceneRenderer fully initialised (%dx%d)", w, h)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-frame rendering
    # ──────────────────────────────────────────────────────────────────────────

    def render(
        self,
        frame_rgb: np.ndarray,
        state: AppState,
        context: StateContext,
        timer: "FrameTimer",
    ) -> None:
        """Execute the full render pipeline for one frame.

        Args:
            frame_rgb: Camera frame as ``(H, W, 3)`` uint8 numpy array (RGB).
            state: Current application state from the FSM.
            context: Mutable state context with interpolated values.
            timer: Frame timer providing *dt*, *elapsed*, and *fps*.
        """
        s = self._settings
        w, h = self._width, self._height
        elapsed = timer.elapsed

        # ── 1. Update background texture ──────────────────────────────────
        if frame_rgb is not None and self._background is not None:
            self._background.update_frame(frame_rgb)

        # ── 2. Clear screen ───────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ── 3. Draw background (camera feed) ─────────────────────────────
        # Renders directly to screen (framebuffer 0)
        if self._background is not None:
            self._background.render()

        # ── 4. 3D scene (if not idle) ─────────────────────────────────────
        if state != AppState.IDLE:
            self._render_3d_scene(state, context, elapsed, w, h)

        # ── 5. HUD overlays ───────────────────────────────────────────────
        if (s.show_fps or self._debug_visible) and self._hud is not None:
            self._render_hud(state, context, timer)

    # ──────────────────────────────────────────────────────────────────────────

    def _render_3d_scene(
        self,
        state: AppState,
        ctx: StateContext,
        elapsed: float,
        width: int,
        height: int,
    ) -> None:
        """Render the holographic 3D scene with bloom post-processing.

        This method:
        1. Begins bloom scene capture (renders into bloom FBO).
        2. Draws the wireframe cube, 3D model, and particles.
        3. Ends capture and applies bloom (extract → blur → combine).
        """
        s = self._settings

        # ── Begin bloom capture ───────────────────────────────────────────
        if self._bloom is not None:
            self._bloom.begin_scene_capture()
            # Clear bloom FBO with transparent black
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Background is drawn directly to screen now to keep camera clear

        # ── Camera / projection matrices ──────────────────────────────────
        aspect = width / max(height, 1)

        # Camera positioned slightly back along +Z
        eye = np.array([0.0, 0.0, 3.0], dtype=np.float32)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        view = pyrr.matrix44.create_look_at(eye, target, up, dtype=np.float32)
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            45.0, aspect, 0.1, 100.0, dtype=np.float32
        )

        # ── Map head_position (normalised 0–1 screen coords) to world ────
        head_x = 0.0
        head_y = 0.0
        if ctx.head_position is not None and len(ctx.head_position) >= 2:
            # NDC: head_position x,y in [0, 1] → remap to [-1.5, 1.5] world range
            head_x = (ctx.head_position[0] - 0.5) * 3.0
            head_y = -(ctx.head_position[1] - 0.5) * 3.0  # Flip Y for OpenGL

        # Cube world position: in front of camera on the XY plane
        cube_world_pos = np.array([head_x, head_y, 0.0], dtype=np.float32)

        # ── Cube model matrix ─────────────────────────────────────────────
        # Translation (head tracking + float offset)
        float_y = cube_world_pos[1] + ctx.cube_float_offset
        translation = pyrr.matrix44.create_from_translation(
            pyrr.Vector3([cube_world_pos[0], float_y, cube_world_pos[2]]),
            dtype=np.float32,
        )

        # Rotation around Y axis
        rotation = pyrr.matrix44.create_from_y_rotation(
            math.radians(ctx.cube_rotation), dtype=np.float32
        )

        # Scale
        cube_size = ctx.cube_scale * s.cube_base_size
        scale = pyrr.matrix44.create_from_scale(
            pyrr.Vector3([cube_size, cube_size, cube_size]),
            dtype=np.float32,
        )

        # Model = Translation × Rotation × Scale  (pyrr uses row-major)
        cube_model = pyrr.matrix44.multiply(scale, rotation)
        cube_model = pyrr.matrix44.multiply(cube_model, translation)

        # ── a. Render wireframe cube ──────────────────────────────────────
        if self._cube is not None:
            self._cube.render(
                model_matrix=cube_model,
                view=view,
                projection=projection,
                opacity=ctx.cube_opacity,
                time=elapsed,
                glow_color=s.glow_color,
                glow_intensity=s.glow_intensity,
            )

        # ── b. Render 3D model (if visible) ───────────────────────────────
        if (
            self._model is not None
            and self._model.has_mesh
            and ctx.model_opacity > 0.0
        ):
            # Model sits inside the cube, independently rotated and scaled
            model_translation = pyrr.matrix44.create_from_translation(
                pyrr.Vector3([cube_world_pos[0], float_y, cube_world_pos[2]]),
                dtype=np.float32,
            )
            model_rotation = pyrr.matrix44.create_from_y_rotation(
                math.radians(ctx.model_rotation), dtype=np.float32
            )

            # Scale the model to fit within the cube (slightly smaller)
            model_size = ctx.model_scale * s.cube_base_size * 0.7
            model_scale = pyrr.matrix44.create_from_scale(
                pyrr.Vector3([model_size, model_size, model_size]),
                dtype=np.float32,
            )

            model_mat = pyrr.matrix44.multiply(model_scale, model_rotation)
            model_mat = pyrr.matrix44.multiply(model_mat, model_translation)

            self._model.render(
                model_matrix=model_mat,
                view=view,
                projection=projection,
                opacity=ctx.model_opacity,
                time=elapsed,
                glow_color=s.glow_color,
                view_pos=eye,
            )

        # ── c. Render particles ───────────────────────────────────────────
        if self._particles is not None:
            particle_radius = cube_size * 0.8 if cube_size > 0.0 else s.cube_base_size * 0.8
            self._particles.render(
                center=np.array(
                    [cube_world_pos[0], float_y, cube_world_pos[2]],
                    dtype=np.float32,
                ),
                radius=particle_radius,
                view=view,
                projection=projection,
                time=elapsed,
                opacity=ctx.cube_opacity * 0.6,  # Particles slightly dimmer than cube
                glow_color=s.glow_color,
            )

        # ── End bloom capture & apply ─────────────────────────────────────
        if self._bloom is not None:
            self._bloom.end_scene_capture()
            self._bloom.apply(
                self._bloom_extract_shader,
                self._bloom_blur_shader,
                self._bloom_combine_shader,
            )

        # Restore clear colour for next frame
        glClearColor(0.0, 0.0, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────────────

    def _render_hud(
        self,
        state: AppState,
        ctx: StateContext,
        timer: "FrameTimer",
    ) -> None:
        """Render the HUD overlay (FPS, state, model name)."""
        if self._hud is None:
            return

        self._hud.begin()

        line_y = 10.0
        line_spacing = 28.0
        text_scale = 0.35

        # FPS counter
        if self._settings.show_fps:
            fps_text = f"FPS: {timer.fps:.0f}"
            self._hud.render_text(fps_text, 10.0, line_y, scale=text_scale, color=(0.0, 1.0, 0.5))
            line_y += line_spacing

        # Debug overlays
        if self._debug_visible:
            # Application state
            state_text = f"State: {state.value}"
            self._hud.render_text(state_text, 10.0, line_y, scale=text_scale, color=(0.0, 0.8, 1.0))
            line_y += line_spacing

            # Cube info
            cube_info = f"Cube: scale={ctx.cube_scale:.2f} rot={ctx.cube_rotation:.1f} opacity={ctx.cube_opacity:.2f}"
            self._hud.render_text(cube_info, 10.0, line_y, scale=text_scale, color=(0.7, 0.7, 0.7))
            line_y += line_spacing

            # Model info
            if ctx.current_model_name:
                model_text = f"Model: {ctx.current_model_name} [{ctx.model_index + 1}/{ctx.total_models}]"
                self._hud.render_text(model_text, 10.0, line_y, scale=text_scale, color=(1.0, 0.8, 0.0))
                line_y += line_spacing

            # Head position
            if ctx.head_position is not None and len(ctx.head_position) >= 2:
                head_text = f"Head: ({ctx.head_position[0]:.2f}, {ctx.head_position[1]:.2f})"
                self._hud.render_text(head_text, 10.0, line_y, scale=text_scale, color=(0.7, 0.7, 0.7))
                line_y += line_spacing

            # Frame timing
            dt_text = f"dt: {timer.dt * 1000.0:.1f}ms  frame: {timer.frame_count}"
            self._hud.render_text(dt_text, 10.0, line_y, scale=text_scale, color=(0.5, 0.5, 0.5))

        self._hud.end()

    # ──────────────────────────────────────────────────────────────────────────
    # Mesh forwarding
    # ──────────────────────────────────────────────────────────────────────────

    def set_mesh(self, mesh: MeshData) -> None:
        """Forward a loaded mesh to the model renderer.

        Args:
            mesh: The :class:`MeshData` to display inside the holographic cube.
        """
        if self._model is not None:
            self._model.set_mesh(mesh)
            logger.info("Scene mesh set: '%s'", mesh.name)
        else:
            logger.warning("Cannot set mesh — ModelRenderer not initialised")

    # ──────────────────────────────────────────────────────────────────────────
    # Resize
    # ──────────────────────────────────────────────────────────────────────────

    def resize(self, width: int, height: int) -> None:
        """Handle window resize — update all subsystems.

        Args:
            width: New window width in pixels.
            height: New window height in pixels.
        """
        self._width = width
        self._height = height
        glViewport(0, 0, width, height)

        if self._hud is not None:
            self._hud.resize(width, height)
        if self._bloom is not None:
            self._bloom.resize(width, height)

        logger.info("SceneRenderer resized to %dx%d", width, height)

    # ──────────────────────────────────────────────────────────────────────────
    # Debug toggle
    # ──────────────────────────────────────────────────────────────────────────

    def toggle_debug(self) -> None:
        """Toggle the visibility of debug HUD overlays."""
        self._debug_visible = not self._debug_visible
        logger.info("Debug HUD %s", "enabled" if self._debug_visible else "disabled")

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release all GPU resources across every subsystem."""
        logger.info("Cleaning up SceneRenderer …")

        # Sub-renderers
        for name, renderer in [
            ("background", self._background),
            ("cube", self._cube),
            ("model", self._model),
            ("particles", self._particles),
            ("hud", self._hud),
        ]:
            if renderer is not None:
                try:
                    renderer.cleanup()
                except Exception:
                    logger.exception("Error cleaning up %s renderer", name)

        # Bloom effect
        if self._bloom is not None:
            try:
                self._bloom.cleanup()
            except Exception:
                logger.exception("Error cleaning up bloom effect")

        # Shaders
        for name, shader in [
            ("background", self._background_shader),
            ("hologram", self._hologram_shader),
            ("model", self._model_shader),
            ("particle", self._particle_shader),
            ("bloom_extract", self._bloom_extract_shader),
            ("bloom_blur", self._bloom_blur_shader),
            ("bloom_combine", self._bloom_combine_shader),
        ]:
            if shader is not None:
                try:
                    shader.delete()
                except Exception:
                    logger.exception("Error deleting %s shader", name)

        # Null out references
        self._background = None
        self._cube = None
        self._model = None
        self._particles = None
        self._hud = None
        self._bloom = None
        self._background_shader = None
        self._hologram_shader = None
        self._model_shader = None
        self._particle_shader = None
        self._bloom_extract_shader = None
        self._bloom_blur_shader = None
        self._bloom_combine_shader = None

        logger.info("SceneRenderer cleanup complete")
