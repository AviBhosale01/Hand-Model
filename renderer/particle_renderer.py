"""
Energy-spark particle renderer for the AR Holographic Gesture Controlled 3D Object Viewer.

Renders a swarm of glowing point-sprite particles that orbit around the
holographic cube.  Particle positions are computed on the CPU each frame using
NumPy vectorised operations for fast parametric orbital motion.

Each particle has:
- A unique orbital angle offset and speed
- A random orbital radius perturbation
- A random height oscillation phase
- A life-cycle value that creates a twinkling / fading effect
"""

import logging
from typing import Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POINTS,
    GL_PROGRAM_POINT_SIZE,
    GL_SRC_ALPHA,
    GL_VERTEX_SHADER,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glBufferSubData,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glVertexAttribPointer,
)

from config.settings import Settings
from graphics.gl_utils import create_vao, create_vbo
from graphics.shader import ShaderProgram

logger = logging.getLogger(__name__)


class ParticleRenderer:
    """Orbit-style energy spark particles around a centre point.

    Particles move in parametric orbits at varying speeds, radii and heights.
    Each particle's *life* cycles between 0 and 1 over time, producing a
    twinkling effect.  All positions are recomputed on the CPU each frame
    using vectorised NumPy arithmetic.

    The VBO layout per particle is ``[x, y, z, life]`` (4 floats = 16 bytes).

    Attributes:
        _shader: Particle point-sprite shader.
        _settings: Application settings.
        _particle_count: Number of particles.
        _angles: Base orbital angle for each particle (radians).
        _speeds: Angular speed multiplier per particle.
        _radii: Per-particle orbital radius offset factor.
        _height_phases: Phase offset for height oscillation.
        _life_phases: Phase offset for the life/twinkling cycle.
        _life_speeds: Speed of the life cycle per particle.
        _vao: Vertex array object.
        _vbo: Vertex buffer object (dynamic).
        _data: Flat ``(N, 4)`` numpy buffer for VBO upload.
    """

    def __init__(self, shader: ShaderProgram, settings: Settings) -> None:
        """Generate initial particle parameters and create GL buffers.

        Args:
            shader: Compiled particle shader program.
            settings: Application settings (provides ``particle_count``).
        """
        self._shader = shader
        self._settings = settings
        self._particle_count: int = max(1, settings.particle_count)

        rng = np.random.default_rng(seed=42)  # deterministic for visual consistency

        # ── Per-particle parameters ───────────────────────────────────────
        n = self._particle_count
        self._angles = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
        self._speeds = rng.uniform(0.3, 1.5, size=n).astype(np.float32)
        self._radii = rng.uniform(0.85, 1.25, size=n).astype(np.float32)
        self._height_phases = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
        self._life_phases = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)
        self._life_speeds = rng.uniform(1.0, 3.5, size=n).astype(np.float32)

        # ── CPU-side data buffer (position + life) ────────────────────────
        self._data = np.zeros((n, 4), dtype=np.float32)

        # ── Create GL resources ───────────────────────────────────────────
        self._vao: int = create_vao()
        glBindVertexArray(self._vao)

        self._vbo: int = create_vbo(self._data.ravel(), usage=GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Attribute 0: vec4 (x, y, z, life)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, False, 4 * 4, None)

        glBindVertexArray(0)

        logger.info("ParticleRenderer initialized: %d particles", n)

    # ──────────────────────────────────────────────────────────────────────────

    def render(
        self,
        center: np.ndarray,
        radius: float,
        view: np.ndarray,
        projection: np.ndarray,
        time: float,
        opacity: float,
        glow_color: Tuple[float, float, float],
    ) -> None:
        """Compute particle positions and draw as point sprites.

        Args:
            center: ``(3,)`` world-space centre of the orbit (cube centre).
            radius: Base orbital radius (should roughly match cube half-extent).
            view: 4×4 view matrix.
            projection: 4×4 projection matrix.
            time: Elapsed application time in seconds.
            opacity: Overall particle opacity in ``[0, 1]``.
            glow_color: RGB glow colour in ``[0, 1]`` range.
        """
        if opacity <= 0.0:
            return

        try:
            # ── Compute positions (vectorised) ────────────────────────────
            angles = self._angles + self._speeds * time
            r = radius * self._radii

            # Orbital X / Z with parametric circle
            self._data[:, 0] = center[0] + r * np.cos(angles)
            self._data[:, 2] = center[2] + r * np.sin(angles)

            # Height oscillation
            self._data[:, 1] = center[1] + 0.3 * radius * np.sin(
                self._height_phases + time * self._speeds * 0.7
            )

            # Life cycle (0 → 1 → 0, repeating) — used for alpha / size
            self._data[:, 3] = (
                0.5 + 0.5 * np.sin(self._life_phases + time * self._life_speeds)
            )

            # ── Upload to GPU ─────────────────────────────────────────────
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, self._data.nbytes, self._data.ravel())

            # ── Shader uniforms ───────────────────────────────────────────
            self._shader.use()
            self._shader.set_mat4("uView", view)
            self._shader.set_mat4("uProjection", projection)
            self._shader.set_vec3("uGlowColor", glow_color)
            self._shader.set_float("uOpacity", opacity)
            self._shader.set_float("uTime", time)

            # ── GL state ──────────────────────────────────────────────────
            glEnable(GL_PROGRAM_POINT_SIZE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending

            # ── Draw ──────────────────────────────────────────────────────
            glBindVertexArray(self._vao)
            glDrawArrays(GL_POINTS, 0, self._particle_count)
            glBindVertexArray(0)

            # ── Restore state ─────────────────────────────────────────────
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_PROGRAM_POINT_SIZE)
        except Exception:
            logger.exception("Error during particle render")
            try:
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDisable(GL_PROGRAM_POINT_SIZE)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release all OpenGL resources held by this renderer."""
        logger.info("Cleaning up ParticleRenderer")
        try:
            if self._vbo:
                glDeleteBuffers(1, [self._vbo])
                self._vbo = 0
        except Exception:
            logger.warning("Failed to delete particle VBO")

        try:
            if self._vao:
                glDeleteVertexArrays(1, [self._vao])
                self._vao = 0
        except Exception:
            logger.warning("Failed to delete particle VAO")
