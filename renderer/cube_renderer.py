"""
Holographic wireframe cube renderer for the AR Holographic Gesture Controlled 3D Object Viewer.

Renders a wireframe cube with glowing edges using ``GL_LINES``.  Additive blending
produces the characteristic holographic glow effect.  The cube is defined as 8
vertices forming 12 edges (24 line-strip indices).
"""

import logging
from typing import Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_LINES,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDrawElements,
    glEnableVertexAttribArray,
    glLineWidth,
    glVertexAttribPointer,
)

from config.settings import Settings
from graphics.gl_utils import create_ebo, create_vao, create_vbo
from graphics.shader import ShaderProgram

logger = logging.getLogger(__name__)

# ── Cube geometry ──────────────────────────────────────────────────────────────
# 8 vertices of a unit cube centred at the origin, in the order:
#   0: (-0.5, -0.5, -0.5)   4: (-0.5,  0.5, -0.5)
#   1: ( 0.5, -0.5, -0.5)   5: ( 0.5,  0.5, -0.5)
#   2: ( 0.5, -0.5,  0.5)   6: ( 0.5,  0.5,  0.5)
#   3: (-0.5, -0.5,  0.5)   7: (-0.5,  0.5,  0.5)

_CUBE_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5],
    ],
    dtype=np.float32,
)

# 12 edges expressed as pairs of vertex indices (24 total for GL_LINES):
# Bottom face, top face, verticals
_CUBE_LINE_INDICES = np.array(
    [
        # Bottom face
        0, 1,  1, 2,  2, 3,  3, 0,
        # Top face
        4, 5,  5, 6,  6, 7,  7, 4,
        # Vertical edges
        0, 4,  1, 5,  2, 6,  3, 7,
    ],
    dtype=np.uint32,
)

_NUM_LINE_INDICES = len(_CUBE_LINE_INDICES)  # 24


class CubeRenderer:
    """Render a holographic wireframe cube with additive-glow edges.

    The cube is a unit cube (edge length 1, centred at origin).  External
    transforms (translation, rotation, scale) are applied via the model matrix
    passed to :meth:`render`.

    Attributes:
        _shader: Hologram shader program for the wireframe cube.
        _settings: Application settings (for defaults / fallback colours).
        _vao: Vertex array object.
        _vbo: Vertex buffer object for the 8 vertices.
        _ebo: Element buffer object for the 24 line indices.
    """

    def __init__(self, shader: ShaderProgram, settings: Settings) -> None:
        """Initialize the wireframe cube geometry and GL buffers.

        Args:
            shader: Compiled hologram shader for wireframe rendering.
            settings: Application-wide settings.
        """
        self._shader = shader
        self._settings = settings

        # Create VAO
        self._vao: int = create_vao()
        glBindVertexArray(self._vao)

        # Create VBO with vertex positions
        self._vbo: int = create_vbo(_CUBE_VERTICES, usage=GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Vertex attribute 0 — vec3 position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0,               # location
            3,               # components per vertex (x, y, z)
            GL_FLOAT,        # data type
            False,           # normalise
            3 * 4,           # stride in bytes (3 floats × 4 bytes)
            None,            # offset
        )

        # Create EBO with line indices
        self._ebo: int = create_ebo(_CUBE_LINE_INDICES, usage=GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)

        # Unbind VAO (keep EBO bound inside it)
        glBindVertexArray(0)

        logger.info(
            "CubeRenderer initialized: VAO=%d, VBO=%d, EBO=%d, indices=%d",
            self._vao,
            self._vbo,
            self._ebo,
            _NUM_LINE_INDICES,
        )

    def render(
        self,
        model_matrix: np.ndarray,
        view: np.ndarray,
        projection: np.ndarray,
        opacity: float,
        time: float,
        glow_color: Tuple[float, float, float],
        glow_intensity: float = 1.5,
    ) -> None:
        """Render the wireframe cube with holographic glow.

        Args:
            model_matrix: 4×4 model (world) transform.
            view: 4×4 view (camera) matrix.
            projection: 4×4 projection matrix.
            opacity: Overall cube opacity in ``[0, 1]``.
            time: Elapsed application time in seconds (used for animation).
            glow_color: RGB glow colour in ``[0, 1]`` range.
            glow_intensity: Multiplicative intensity for the glow (default 1.5).
        """
        if opacity <= 0.0:
            return

        try:
            self._shader.use()

            # ── Set uniforms ──────────────────────────────────────────────
            self._shader.set_mat4("uModel", model_matrix)
            self._shader.set_mat4("uView", view)
            self._shader.set_mat4("uProjection", projection)
            self._shader.set_vec3("uGlowColor", glow_color)
            self._shader.set_float("uOpacity", opacity)
            self._shader.set_float("uTime", time)
            self._shader.set_float("uGlowIntensity", glow_intensity)

            # ── GL state for additive glow ────────────────────────────────
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)
            glLineWidth(2.0)

            # ── Draw ──────────────────────────────────────────────────────
            glBindVertexArray(self._vao)
            glDrawElements(GL_LINES, _NUM_LINE_INDICES, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

            # ── Restore default blend mode ────────────────────────────────
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        except Exception:
            logger.exception("Error during cube render")
            # Attempt to restore blend state on failure
            try:
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            except Exception:
                pass

    def cleanup(self) -> None:
        """Release all OpenGL resources held by this renderer."""
        logger.info("Cleaning up CubeRenderer")
        try:
            if self._ebo:
                glDeleteBuffers(1, [self._ebo])
                self._ebo = 0
        except Exception:
            logger.warning("Failed to delete cube EBO")

        try:
            if self._vbo:
                glDeleteBuffers(1, [self._vbo])
                self._vbo = 0
        except Exception:
            logger.warning("Failed to delete cube VBO")

        try:
            if self._vao:
                glDeleteVertexArrays(1, [self._vao])
                self._vao = 0
        except Exception:
            logger.warning("Failed to delete cube VAO")
