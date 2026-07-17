"""
Holographic 3D model renderer for the AR Holographic Gesture Controlled 3D Object Viewer.

Renders loaded 3D meshes (:class:`MeshData`) with a holographic shader aesthetic.
Vertex data is interleaved as ``[px, py, pz, nx, ny, nz]`` for each vertex,
enabling both position-based transforms and normal-based lighting / fresnel effects.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_DEPTH_TEST,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_ONE,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_STATIC_DRAW,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNSIGNED_INT,
    glBindBuffer,
    glBindVertexArray,
    glBlendFunc,
    glDeleteBuffers,
    glDeleteVertexArrays,
    glDepthMask,
    glDrawElements,
    glEnableVertexAttribArray,
    glVertexAttribPointer,
)
from OpenGL.GL import ctypes as gl_ctypes

from config.settings import Settings
from graphics.gl_utils import create_ebo, create_vao, create_vbo
from graphics.shader import ShaderProgram
from models.loader import MeshData

logger = logging.getLogger(__name__)

# Stride for interleaved vertex data: position(3) + normal(3) + color(3) = 9 floats = 36 bytes
_VERTEX_STRIDE = 9 * 4
_NORMAL_OFFSET = 3 * 4   # 12 bytes (offset to the normal component)
_COLOR_OFFSET = 6 * 4    # 24 bytes (offset to the color component)


class ModelRenderer:
    """Render 3D meshes with a holographic visual style.

    Mesh data is interleaved into a single VBO with layout
    ``[px, py, pz, nx, ny, nz]`` per vertex, supporting both positional
    transforms and normal-based effects (fresnel rim-glow, edge highlighting).

    The renderer uses additive blending and writes no depth so the holographic
    model composites cleanly over the wireframe cube and background.

    Attributes:
        _shader: Model holographic shader program.
        _settings: Application settings.
        _vao: Current vertex array object (or ``None`` if no mesh loaded).
        _vbo: Current vertex buffer object.
        _ebo: Current element buffer object.
        _current_mesh: The currently loaded :class:`MeshData` (or ``None``).
        _index_count: Number of indices in the active EBO.
    """

    def __init__(self, shader: ShaderProgram, settings: Settings) -> None:
        """Initialize the model renderer (no mesh loaded yet).

        Args:
            shader: Compiled model holographic shader.
            settings: Application-wide settings.
        """
        self._shader = shader
        self._settings = settings

        self._vao: Optional[int] = None
        self._vbo: Optional[int] = None
        self._ebo: Optional[int] = None
        self._current_mesh: Optional[MeshData] = None
        self._index_count: int = 0

        logger.info("ModelRenderer initialized (no mesh loaded)")

    # ──────────────────────────────────────────────────────────────────────────
    # Mesh management
    # ──────────────────────────────────────────────────────────────────────────

    def set_mesh(self, mesh: MeshData) -> None:
        """Load a new mesh, replacing any previously loaded geometry.

        Interleaves vertices and normals into a single VBO and builds the
        corresponding EBO.

        Args:
            mesh: The :class:`MeshData` to load.
        """
        # Tear down previous GPU resources
        self._release_buffers()

        self._current_mesh = mesh
        self._index_count = mesh.index_count

        # ── Interleave vertex data ────────────────────────────────────────
        # Resulting shape: (N, 9) — [px, py, pz, nx, ny, nz, cx, cy, cz]
        try:
            interleaved = np.empty((mesh.vertex_count, 9), dtype=np.float32)
            interleaved[:, 0:3] = mesh.vertices[:mesh.vertex_count]
            interleaved[:, 3:6] = mesh.normals[:mesh.vertex_count]
            interleaved[:, 6:9] = mesh.colors[:mesh.vertex_count]
        except Exception:
            logger.exception("Failed to interleave vertex data for mesh '%s'", mesh.name)
            raise

        # ── Create VAO ────────────────────────────────────────────────────
        self._vao = create_vao()
        glBindVertexArray(self._vao)

        # ── Create VBO ────────────────────────────────────────────────────
        self._vbo = create_vbo(interleaved.ravel(), usage=GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Attribute 0: position (vec3), stride = 36, offset = 0
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, False,
            _VERTEX_STRIDE,
            None,
        )

        # Attribute 1: normal (vec3), stride = 36, offset = 12
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            1, 3, GL_FLOAT, False,
            _VERTEX_STRIDE,
            gl_ctypes.c_void_p(_NORMAL_OFFSET),
        )

        # Attribute 2: color (vec3), stride = 36, offset = 24
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(
            2, 3, GL_FLOAT, False,
            _VERTEX_STRIDE,
            gl_ctypes.c_void_p(_COLOR_OFFSET),
        )

        # ── Create EBO ────────────────────────────────────────────────────
        self._ebo = create_ebo(mesh.indices, usage=GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)

        glBindVertexArray(0)

        logger.info(
            "Mesh '%s' loaded: verts=%d, indices=%d, VAO=%d",
            mesh.name,
            mesh.vertex_count,
            mesh.index_count,
            self._vao,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Rendering
    # ──────────────────────────────────────────────────────────────────────────

    def render(
        self,
        model_matrix: np.ndarray,
        view: np.ndarray,
        projection: np.ndarray,
        opacity: float,
        time: float,
        glow_color: Tuple[float, float, float],
        view_pos: Optional[np.ndarray] = None,
    ) -> None:
        """Render the loaded mesh with the holographic shader.

        Args:
            model_matrix: 4×4 model (world) transform.
            view: 4×4 view (camera) matrix.
            projection: 4×4 projection matrix.
            opacity: Overall model opacity in ``[0, 1]``.
            time: Elapsed application time in seconds.
            glow_color: RGB glow colour in ``[0, 1]`` range.
            view_pos: Camera world-space position (vec3).  Falls back to
                      ``(0, 0, 3)`` if not supplied.
        """
        if not self.has_mesh or opacity <= 0.0:
            return

        if view_pos is None:
            view_pos = np.array([0.0, 0.0, 3.0], dtype=np.float32)

        try:
            self._shader.use()

            # ── Compute normal matrix: transpose(inverse(mat3(model))) ────
            model_3x3 = model_matrix[:3, :3].astype(np.float64)
            try:
                normal_matrix = np.linalg.inv(model_3x3).T.astype(np.float32)
            except np.linalg.LinAlgError:
                # Fallback to identity if the model matrix is singular
                normal_matrix = np.eye(3, dtype=np.float32)
                logger.warning("Singular model matrix; using identity normal matrix")

            # Pad the 3×3 normal matrix to 4×4 for uniform upload
            normal_mat4 = np.eye(4, dtype=np.float32)
            normal_mat4[:3, :3] = normal_matrix

            # ── Set uniforms ──────────────────────────────────────────────
            self._shader.set_mat4("uModel", model_matrix)
            self._shader.set_mat4("uView", view)
            self._shader.set_mat4("uProjection", projection)
            self._shader.set_mat4("uNormalMatrix", normal_mat4)
            self._shader.set_vec3("uGlowColor", glow_color)
            self._shader.set_float("uOpacity", opacity)
            self._shader.set_float("uTime", time)
            self._shader.set_vec3("uViewPos", view_pos)

            # ── GL state for holographic look ─────────────────────────────
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)   # Additive blending
            glDepthMask(GL_FALSE)               # Don't write to depth buffer

            # ── Draw ──────────────────────────────────────────────────────
            glBindVertexArray(self._vao)
            glDrawElements(GL_TRIANGLES, self._index_count, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)

            # ── Restore state ─────────────────────────────────────────────
            glDepthMask(GL_TRUE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        except Exception:
            logger.exception("Error during model render")
            # Best-effort state restoration
            try:
                glDepthMask(GL_TRUE)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def has_mesh(self) -> bool:
        """Return ``True`` if a mesh is currently loaded and ready to render."""
        return self._current_mesh is not None and self._vao is not None

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def _release_buffers(self) -> None:
        """Delete current VAO/VBO/EBO if they exist."""
        if self._ebo is not None:
            try:
                glDeleteBuffers(1, [self._ebo])
            except Exception:
                logger.warning("Failed to delete model EBO")
            self._ebo = None

        if self._vbo is not None:
            try:
                glDeleteBuffers(1, [self._vbo])
            except Exception:
                logger.warning("Failed to delete model VBO")
            self._vbo = None

        if self._vao is not None:
            try:
                glDeleteVertexArrays(1, [self._vao])
            except Exception:
                logger.warning("Failed to delete model VAO")
            self._vao = None

    def cleanup(self) -> None:
        """Release all OpenGL resources held by this renderer."""
        logger.info("Cleaning up ModelRenderer")
        self._release_buffers()
        self._current_mesh = None
        self._index_count = 0
