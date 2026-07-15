"""OpenGL utility functions for buffer, texture, and framebuffer management.

Provides convenience functions for creating and destroying VAOs, VBOs, EBOs,
textures, framebuffers, and a fullscreen quad geometry.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_R8,
    GL_RED,
    GL_RGB,
    GL_RGB16F,
    GL_RGBA,
    GL_RGBA16F,
    GL_STATIC_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBindBuffer,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBufferData,
    glCheckFramebufferStatus,
    glDeleteBuffers,
    glDeleteFramebuffers,
    glDeleteTextures,
    glDeleteVertexArrays,
    glEnableVertexAttribArray,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glVertexAttribPointer,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

# Map internal formats to the appropriate (format, type) pair for glTexImage2D
_FORMAT_MAP = {
    GL_RGB:      (GL_RGB,  GL_UNSIGNED_BYTE),
    GL_RGBA:     (GL_RGBA, GL_UNSIGNED_BYTE),
    GL_R8:       (GL_RED,  GL_UNSIGNED_BYTE),
    GL_RGB16F:   (GL_RGB,  GL_FLOAT),
    GL_RGBA16F:  (GL_RGBA, GL_FLOAT),
}


def _resolve_format(internal_format: int) -> Tuple[int, int]:
    """Resolve an internal format to its (format, type) pair.

    Args:
        internal_format: The OpenGL internal format constant.

    Returns:
        A (format, type) tuple suitable for glTexImage2D / glTexSubImage2D.
    """
    if internal_format in _FORMAT_MAP:
        return _FORMAT_MAP[internal_format]
    # Fallback: assume RGBA unsigned byte
    logger.warning(
        "Unknown internal format 0x%X; falling back to GL_RGBA / GL_UNSIGNED_BYTE",
        internal_format,
    )
    return (GL_RGBA, GL_UNSIGNED_BYTE)


# ─────────────────────────────────────────────────────────────────────────────
# VAO / VBO / EBO creation
# ─────────────────────────────────────────────────────────────────────────────


def create_vao() -> int:
    """Create and return a new Vertex Array Object (VAO).

    Returns:
        The VAO handle.
    """
    vao = glGenVertexArrays(1)
    logger.debug("Created VAO (id=%d)", vao)
    return int(vao)


def create_vbo(data: np.ndarray, usage: int = GL_STATIC_DRAW) -> int:
    """Create a Vertex Buffer Object (VBO) and upload data.

    Args:
        data: Vertex data as a numpy array. Will be converted to float32
              and made contiguous if necessary.
        usage: OpenGL buffer usage hint (e.g. GL_STATIC_DRAW).

    Returns:
        The VBO handle.
    """
    arr = np.ascontiguousarray(data, dtype=np.float32)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, int(vbo))
    glBufferData(GL_ARRAY_BUFFER, arr.nbytes, arr, usage)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    logger.debug("Created VBO (id=%d, %d bytes)", vbo, arr.nbytes)
    return int(vbo)


def create_ebo(data: np.ndarray, usage: int = GL_STATIC_DRAW) -> int:
    """Create an Element Buffer Object (EBO) and upload index data.

    Args:
        data: Index data as a numpy array. Will be converted to uint32
              and made contiguous if necessary.
        usage: OpenGL buffer usage hint.

    Returns:
        The EBO handle.
    """
    arr = np.ascontiguousarray(data, dtype=np.uint32)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, int(ebo))
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, arr.nbytes, arr, usage)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    logger.debug("Created EBO (id=%d, %d bytes)", ebo, arr.nbytes)
    return int(ebo)


# ─────────────────────────────────────────────────────────────────────────────
# Texture utilities
# ─────────────────────────────────────────────────────────────────────────────


def create_texture_2d(
    width: int,
    height: int,
    internal_format: int = GL_RGB,
    data: Optional[np.ndarray] = None,
) -> int:
    """Create a 2D texture with optional initial data.

    Sets filtering to GL_LINEAR and wrap mode to GL_CLAMP_TO_EDGE.

    Args:
        width: Texture width in pixels.
        height: Texture height in pixels.
        internal_format: OpenGL internal format (e.g. GL_RGB, GL_RGBA16F).
        data: Optional numpy array of pixel data. Pass None for an empty texture.

    Returns:
        The texture handle.
    """
    fmt, dtype = _resolve_format(internal_format)

    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, int(texture))

    # Upload or allocate
    pixel_data = None
    if data is not None:
        pixel_data = np.ascontiguousarray(data)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        internal_format,
        width,
        height,
        0,
        fmt,
        dtype,
        pixel_data,
    )

    # Filtering and wrap
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    glBindTexture(GL_TEXTURE_2D, 0)
    logger.debug(
        "Created texture 2D (id=%d, %dx%d, format=0x%X)",
        texture,
        width,
        height,
        internal_format,
    )
    return int(texture)


def update_texture_2d(
    texture_id: int,
    width: int,
    height: int,
    data: np.ndarray,
    fmt: int = GL_RGB,
) -> None:
    """Update an existing 2D texture with new pixel data.

    Args:
        texture_id: The texture to update.
        width: Width of the new data in pixels.
        height: Height of the new data in pixels.
        data: Pixel data as a contiguous numpy array.
        fmt: Pixel data format (e.g. GL_RGB, GL_RGBA).
    """
    pixel_data = np.ascontiguousarray(data)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexSubImage2D(
        GL_TEXTURE_2D,
        0,   # mip level
        0,   # x offset
        0,   # y offset
        width,
        height,
        fmt,
        GL_UNSIGNED_BYTE,
        pixel_data,
    )
    glBindTexture(GL_TEXTURE_2D, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Framebuffer utilities
# ─────────────────────────────────────────────────────────────────────────────


def create_fbo_with_texture(
    width: int,
    height: int,
    internal_format: int = GL_RGBA16F,
) -> Tuple[int, int]:
    """Create a Framebuffer Object (FBO) with an attached color texture.

    Args:
        width: Framebuffer width in pixels.
        height: Framebuffer height in pixels.
        internal_format: Internal format for the color attachment texture.

    Returns:
        A (fbo, texture) tuple of OpenGL handles.

    Raises:
        RuntimeError: If the framebuffer is incomplete.
    """
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, int(fbo))

    # Create and attach color texture
    texture = create_texture_2d(width, height, internal_format)
    glBindTexture(GL_TEXTURE_2D, texture)
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        texture,
        0,  # mip level
    )

    # Check completeness
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        # Clean up on failure
        glDeleteTextures(1, [texture])
        glDeleteFramebuffers(1, [int(fbo)])
        raise RuntimeError(
            f"Framebuffer incomplete (status=0x{status:X}). "
            f"Requested {width}x{height} with format 0x{internal_format:X}"
        )

    glBindTexture(GL_TEXTURE_2D, 0)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    logger.debug(
        "Created FBO (id=%d) with color texture (id=%d, %dx%d)",
        fbo,
        texture,
        width,
        height,
    )
    return (int(fbo), texture)


# ─────────────────────────────────────────────────────────────────────────────
# Fullscreen quad geometry
# ─────────────────────────────────────────────────────────────────────────────


def create_fullscreen_quad_vao() -> Tuple[int, int]:
    """Create a VAO and VBO for a standard fullscreen quad covering [-1, 1] NDC.
    Used for post-processing where standard OpenGL UV orientation is expected.
    """
    # fmt: off
    vertices = np.array([
        # position (x, y)   texcoord (u, v)
        -1.0, -1.0,         0.0, 0.0,   # bottom-left
         1.0, -1.0,         1.0, 0.0,   # bottom-right
         1.0,  1.0,         1.0, 1.0,   # top-right

        -1.0, -1.0,         0.0, 0.0,   # bottom-left
         1.0,  1.0,         1.0, 1.0,   # top-right
        -1.0,  1.0,         0.0, 1.0,   # top-left
    ], dtype=np.float32)
    # fmt: on

    stride = 4 * vertices.itemsize

    vao = create_vao()
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, int(vbo))
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Attribute 0: position (vec2)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, False, stride, None)

    # Attribute 1: texcoord (vec2)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(
        1, 2, GL_FLOAT, False, stride, ctypes.c_void_p(2 * vertices.itemsize)
    )

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    vbo_int = int(vbo)
    logger.debug("Created standard fullscreen quad (vao=%d, vbo=%d)", vao, vbo_int)
    return (vao, vbo_int)


def create_flipped_fullscreen_quad_vao() -> Tuple[int, int]:
    """Create a VAO and VBO for a vertically flipped fullscreen quad covering [-1, 1] NDC.
    Used for drawing OpenCV camera textures where the Y coordinate is inverted relative to OpenGL.
    """
    # fmt: off
    vertices = np.array([
        # position (x, y)   texcoord (u, v)
        -1.0, -1.0,         0.0, 1.0,   # bottom-left
         1.0, -1.0,         1.0, 1.0,   # bottom-right
         1.0,  1.0,         1.0, 0.0,   # top-right

        -1.0, -1.0,         0.0, 1.0,   # bottom-left
         1.0,  1.0,         1.0, 0.0,   # top-right
        -1.0,  1.0,         0.0, 0.0,   # top-left
    ], dtype=np.float32)
    # fmt: on

    stride = 4 * vertices.itemsize

    vao = create_vao()
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, int(vbo))
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Attribute 0: position (vec2)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 2, GL_FLOAT, False, stride, None)

    # Attribute 1: texcoord (vec2)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(
        1, 2, GL_FLOAT, False, stride, ctypes.c_void_p(2 * vertices.itemsize)
    )

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    vbo_int = int(vbo)
    logger.debug("Created flipped fullscreen quad (vao=%d, vbo=%d)", vao, vbo_int)
    return (vao, vbo_int)


# ─────────────────────────────────────────────────────────────────────────────
# Resource deletion
# ─────────────────────────────────────────────────────────────────────────────


def delete_vao(vao: int) -> None:
    """Delete a Vertex Array Object.

    Args:
        vao: The VAO handle to delete. Ignored if <= 0.
    """
    if vao > 0:
        glDeleteVertexArrays(1, [vao])
        logger.debug("Deleted VAO (id=%d)", vao)


def delete_vbo(vbo: int) -> None:
    """Delete a buffer object (VBO or EBO).

    Args:
        vbo: The buffer handle to delete. Ignored if <= 0.
    """
    if vbo > 0:
        glDeleteBuffers(1, [vbo])
        logger.debug("Deleted buffer (id=%d)", vbo)


def delete_texture(tex: int) -> None:
    """Delete a texture object.

    Args:
        tex: The texture handle to delete. Ignored if <= 0.
    """
    if tex > 0:
        glDeleteTextures(1, [tex])
        logger.debug("Deleted texture (id=%d)", tex)


def delete_fbo(fbo: int) -> None:
    """Delete a Framebuffer Object.

    Args:
        fbo: The FBO handle to delete. Ignored if <= 0.
    """
    if fbo > 0:
        glDeleteFramebuffers(1, [fbo])
        logger.debug("Deleted FBO (id=%d)", fbo)


# Required for ctypes.c_void_p used in vertex attribute offsets
import ctypes  # noqa: E402 — imported at end to keep primary imports clean
