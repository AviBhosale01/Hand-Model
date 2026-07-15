"""
Background renderer for the AR Holographic Gesture Controlled 3D Object Viewer.

Renders the live camera feed as a fullscreen background quad behind all 3D content.
Uses a streaming texture updated each frame with the latest camera RGB data.
"""

import logging
from typing import Optional

import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FALSE,
    GL_RGB,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TRIANGLES,
    GL_TRUE,
    glActiveTexture,
    glBindTexture,
    glBindVertexArray,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glEnable,
)

from graphics.gl_utils import create_flipped_fullscreen_quad_vao, create_texture_2d, update_texture_2d
from graphics.shader import ShaderProgram

logger = logging.getLogger(__name__)


class BackgroundRenderer:
    """Renders the camera feed as a fullscreen background behind all 3D content.

    The renderer maintains a streaming GL texture that is updated each frame
    with the latest camera RGB data via ``glTexSubImage2D`` for efficiency.
    The texture is drawn onto a fullscreen quad with depth testing disabled
    so that it always appears behind all 3D scene elements.

    Attributes:
        _shader: The background shader program (simple textured quad).
        _width: Width of the camera frame in pixels.
        _height: Height of the camera frame in pixels.
        _quad_vao: Vertex array object for the fullscreen quad.
        _quad_vbo: Vertex buffer object for the fullscreen quad.
        _texture_id: OpenGL texture handle for the camera frame.
        _has_frame: Whether at least one frame has been uploaded.
    """

    def __init__(self, shader: ShaderProgram, width: int, height: int) -> None:
        """Initialize the background renderer.

        Args:
            shader: Compiled shader program for rendering the background quad.
            width: Width of the camera frame in pixels.
            height: Height of the camera frame in pixels.
        """
        self._shader = shader
        self._width = width
        self._height = height
        self._has_frame: bool = False

        # Create the fullscreen quad geometry
        try:
            self._quad_vao, self._quad_vbo = create_flipped_fullscreen_quad_vao()
            logger.debug("Background fullscreen quad VAO created: %d", self._quad_vao)
        except Exception:
            logger.exception("Failed to create fullscreen quad VAO")
            raise

        # Create a streaming texture for the camera frames
        try:
            self._texture_id: int = create_texture_2d(width, height, internal_format=GL_RGB)
            logger.debug(
                "Background texture created: id=%d, size=%dx%d",
                self._texture_id,
                width,
                height,
            )
        except Exception:
            logger.exception("Failed to create background texture")
            raise

        logger.info("BackgroundRenderer initialized (%dx%d)", width, height)

    def update_frame(self, frame_rgb: np.ndarray) -> None:
        """Update the GL texture with a new camera frame.

        Uses ``update_texture_2d`` (backed by ``glTexSubImage2D``) for efficient
        streaming updates without reallocating the texture each frame.

        Args:
            frame_rgb: Camera frame as an ``(H, W, 3)`` uint8 numpy array in RGB order.
        """
        if frame_rgb is None:
            return

        h, w = frame_rgb.shape[:2]

        # If frame dimensions changed, recreate the texture
        if w != self._width or h != self._height:
            logger.info(
                "Camera frame size changed from %dx%d to %dx%d — recreating texture",
                self._width,
                self._height,
                w,
                h,
            )
            self._width = w
            self._height = h
            try:
                glDeleteTextures([self._texture_id])
            except Exception:
                logger.warning("Failed to delete old background texture during resize")
            self._texture_id = create_texture_2d(w, h, internal_format=GL_RGB)

        try:
            update_texture_2d(self._texture_id, self._width, self._height, frame_rgb, fmt=GL_RGB)
            self._has_frame = True
        except Exception:
            logger.exception("Failed to update background texture")

    def render(self) -> None:
        """Draw the camera feed as a fullscreen background.

        Disables depth testing so the quad is always behind everything,
        then re-enables it after drawing.
        """
        if not self._has_frame:
            return

        try:
            # Disable depth test so quad appears behind everything
            glDisable(GL_DEPTH_TEST)

            self._shader.use()
            self._shader.set_int("uTexture", 0)

            # Bind camera texture to unit 0
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._texture_id)

            # Draw the fullscreen quad (6 vertices: 2 triangles)
            glBindVertexArray(self._quad_vao)
            glDrawArrays(GL_TRIANGLES, 0, 6)
            glBindVertexArray(0)

            # Re-enable depth test for subsequent 3D rendering
            glEnable(GL_DEPTH_TEST)
        except Exception:
            logger.exception("Error during background render")
            # Ensure depth test is restored even on failure
            glEnable(GL_DEPTH_TEST)

    def cleanup(self) -> None:
        """Release all OpenGL resources held by this renderer."""
        logger.info("Cleaning up BackgroundRenderer")
        try:
            if self._texture_id:
                glDeleteTextures([self._texture_id])
                self._texture_id = 0
        except Exception:
            logger.warning("Failed to delete background texture")

        try:
            if self._quad_vao:
                glDeleteVertexArrays(1, [self._quad_vao])
                self._quad_vao = 0
        except Exception:
            logger.warning("Failed to delete background quad VAO")
