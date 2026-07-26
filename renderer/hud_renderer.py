"""
HUD (Heads-Up Display) renderer for the AR Holographic Gesture Controlled 3D Object Viewer.

Renders on-screen text overlays (FPS counter, state info, debug data) using a
bitmap font atlas generated at startup with Pillow.  All text is drawn as
textured quads in an orthographic projection overlaid on the scene.

The font atlas is a 512×512 image containing printable ASCII characters (32–126)
arranged in a grid.  Character UV coordinates are precomputed and stored in a
lookup table for fast quad generation.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_DEPTH_TEST,
    GL_DYNAMIC_DRAW,
    GL_FLOAT,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RED,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNPACK_ALIGNMENT,
    glActiveTexture,
    glBindBuffer,
    glBindTexture,
    glBindVertexArray,
    glBlendFunc,
    glBufferData,
    glBufferSubData,
    glDeleteBuffers,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDisable,
    glDrawArrays,
    glEnable,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenTextures,
    glGenVertexArrays,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri,
    glVertexAttribPointer,
)
from OpenGL.GL import ctypes as gl_ctypes

from graphics.shader import ShaderProgram

logger = logging.getLogger(__name__)

# ── Atlas configuration ────────────────────────────────────────────────────────
_ATLAS_SIZE = 512        # Width and height of the font atlas texture
_CHAR_START = 32         # First printable ASCII char (space)
_CHAR_END = 126          # Last printable ASCII char (~)
_CHAR_COUNT = _CHAR_END - _CHAR_START + 1  # 95 characters
_GRID_COLS = 16          # Characters per row in the atlas
_GRID_ROWS = 6           # Rows in the atlas (ceil(95/16)=6)
_CELL_W = _ATLAS_SIZE // _GRID_COLS   # 32 px
_CELL_H = _ATLAS_SIZE // _GRID_ROWS   # ~85 px
_MAX_TEXT_CHARS = 512    # Maximum characters per draw call
_VERTS_PER_CHAR = 6      # 2 triangles × 3 vertices
_FLOATS_PER_VERT = 8     # x, y, z, w(unused), u, v, _, _ → simplified to x,y,u,v


class HUDRenderer:
    """On-screen text overlay renderer using a bitmap font atlas.

    Generates a monospace font atlas at startup using Pillow and renders
    arbitrary ASCII text as textured quads.  The renderer operates in an
    orthographic projection so text is screen-aligned regardless of the 3D
    camera.

    Attributes:
        _width: Screen width in pixels.
        _height: Screen height in pixels.
        _atlas_texture: GL texture handle for the font atlas.
        _char_uvs: Mapping from character ordinal to ``(u0, v0, u1, v1)``.
        _char_width: On-screen width of each character cell (pre-scale).
        _char_height: On-screen height of each character cell (pre-scale).
        _vao: Vertex array object for text quads.
        _vbo: Dynamic vertex buffer object.
        _shader: HUD text shader program.
        _projection: 4×4 orthographic projection matrix.
    """

    def __init__(self, width: int, height: int) -> None:
        """Generate the font atlas and create GL resources.

        Args:
            width: Screen width in pixels.
            height: Screen height in pixels.
        """
        self._width = width
        self._height = height

        # ── Generate font atlas ───────────────────────────────────────────
        self._char_uvs: Dict[int, Tuple[float, float, float, float]] = {}
        self._char_width: int = _CELL_W
        self._char_height: int = _CELL_H
        atlas_data = self._generate_atlas()

        # ── Upload atlas to GPU ───────────────────────────────────────────
        self._atlas_texture = self._create_atlas_texture(atlas_data)
        self._white_texture = self._create_white_texture()

        # ── Create dynamic VBO for text quads ─────────────────────────────
        self._vao = int(glGenVertexArrays(1))
        glBindVertexArray(self._vao)

        self._vbo = int(glGenBuffers(1))
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)

        # Pre-allocate buffer: max chars × 6 verts × 4 floats (x, y, u, v)
        max_bytes = _MAX_TEXT_CHARS * _VERTS_PER_CHAR * 4 * 4  # 4 floats × 4 bytes
        glBufferData(GL_ARRAY_BUFFER, max_bytes, None, GL_DYNAMIC_DRAW)

        # Attribute 0: vec2 position (x, y)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 4 * 4, None)

        # Attribute 1: vec2 texcoord (u, v)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, 4 * 4, gl_ctypes.c_void_p(2 * 4))

        glBindVertexArray(0)

        # ── Load HUD shader ──────────────────────────────────────────────
        try:
            self._shader = ShaderProgram("shaders/hud.vert", "shaders/hud.frag")
        except Exception:
            logger.exception("Failed to load HUD shaders")
            raise

        # ── Orthographic projection ───────────────────────────────────────
        self._projection = self._ortho_projection(width, height)

        logger.info("HUDRenderer initialized (%dx%d), atlas=%d", width, height, self._atlas_texture)

    # ──────────────────────────────────────────────────────────────────────────
    # Atlas generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_atlas(self) -> np.ndarray:
        """Generate a grayscale font atlas image using Pillow.

        Returns:
            A ``(512, 512)`` uint8 numpy array with white-on-black glyphs.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.error("Pillow is required for HUD font atlas generation")
            raise

        img = Image.new("L", (_ATLAS_SIZE, _ATLAS_SIZE), color=0)
        draw = ImageDraw.Draw(img)

        # Try to load a monospace font; fall back to Pillow's default
        font: Optional[ImageFont.FreeTypeFont] = None
        font_size = int(_CELL_H * 0.7)
        font_candidates = [
            "consola.ttf",          # Windows Consolas
            "cour.ttf",             # Windows Courier New
            "DejaVuSansMono.ttf",   # Linux
            "LiberationMono-Regular.ttf",
        ]
        for font_name in font_candidates:
            try:
                font = ImageFont.truetype(font_name, font_size)
                logger.debug("Loaded font: %s (size %d)", font_name, font_size)
                break
            except (IOError, OSError):
                continue

        if font is None:
            logger.warning("No TrueType monospace font found; using Pillow default")
            font = ImageFont.load_default()

        # Render each printable ASCII character into the grid
        for i in range(_CHAR_COUNT):
            char = chr(_CHAR_START + i)
            col = i % _GRID_COLS
            row = i // _GRID_COLS

            x = col * _CELL_W
            y = row * _CELL_H

            # Centre the glyph within the cell
            try:
                bbox = font.getbbox(char)
                gw = bbox[2] - bbox[0]
                gh = bbox[3] - bbox[1]
            except AttributeError:
                # Older Pillow without getbbox
                gw, gh = draw.textsize(char, font=font)

            cx = x + (_CELL_W - gw) // 2
            cy = y + (_CELL_H - gh) // 2

            draw.text((cx, cy), char, fill=255, font=font)

            # Store normalised UV coordinates for this character
            u0 = x / _ATLAS_SIZE
            v0 = y / _ATLAS_SIZE
            u1 = (x + _CELL_W) / _ATLAS_SIZE
            v1 = (y + _CELL_H) / _ATLAS_SIZE
            self._char_uvs[_CHAR_START + i] = (u0, v0, u1, v1)

        atlas_data = np.array(img, dtype=np.uint8)
        logger.debug("Font atlas generated: %d characters in %dx%d grid", _CHAR_COUNT, _GRID_COLS, _GRID_ROWS)
        return atlas_data

    def _create_atlas_texture(self, atlas_data: np.ndarray) -> int:
        """Upload the font atlas to an OpenGL texture.

        Args:
            atlas_data: ``(512, 512)`` uint8 grayscale image.

        Returns:
            GL texture handle.
        """
        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RED,
            _ATLAS_SIZE, _ATLAS_SIZE, 0,
            GL_RED, GL_FLOAT if atlas_data.dtype == np.float32 else 0x1401,  # GL_UNSIGNED_BYTE
            atlas_data,
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    def _create_white_texture(self) -> int:
        """Create a 1x1 white texture for drawing solid colored UI quads."""
        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)
        white_pixel = np.array([255], dtype=np.uint8)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 1, 1, 0, GL_RED, 0x1401, white_pixel)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    # ──────────────────────────────────────────────────────────────────────────
    # Orthographic projection
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _ortho_projection(width: int, height: int) -> np.ndarray:
        """Create a 2D orthographic projection matrix.

        Maps screen coordinates ``(0..width, 0..height)`` to clip space.
        Origin is at the top-left.

        Args:
            width: Screen width.
            height: Screen height.

        Returns:
            4×4 float32 projection matrix.
        """
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 / width
        proj[1, 1] = -2.0 / height   # Y-axis flipped (top-left origin)
        proj[2, 2] = -1.0
        proj[3, 3] = 1.0
        proj[3, 0] = -1.0
        proj[3, 1] = 1.0
        return proj

    # ──────────────────────────────────────────────────────────────────────────
    # Text rendering
    # ──────────────────────────────────────────────────────────────────────────

    def render_text(
        self,
        text: str,
        x: float,
        y: float,
        scale: float = 1.0,
        color: Tuple[float, float, float] = (0.0, 0.8, 1.0),
    ) -> None:
        """Render a string of ASCII text at the given screen position.

        Args:
            text: The text string to render (printable ASCII only).
            x: X position in pixels (top-left origin).
            y: Y position in pixels (top-left origin).
            scale: Uniform scale factor (1.0 = native atlas cell size).
            color: RGB text colour in ``[0, 1]`` range.
        """
        if not text:
            return

        # Clamp to max characters
        text = text[:_MAX_TEXT_CHARS]

        cw = self._char_width * scale
        ch = self._char_height * scale

        # ── Build quad vertices for each character ────────────────────────
        vertices = []
        cursor_x = x
        for char in text:
            code = ord(char)
            if code < _CHAR_START or code > _CHAR_END:
                # Non-printable — advance cursor but don't draw
                cursor_x += cw
                continue

            u0, v0, u1, v1 = self._char_uvs[code]

            # Two triangles for the character quad:
            #   (cursor_x, y) ──── (cursor_x+cw, y)
            #        |                     |
            #   (cursor_x, y+ch) ─ (cursor_x+cw, y+ch)
            x0, y0 = cursor_x, y
            x1, y1 = cursor_x + cw, y + ch

            # Triangle 1
            vertices.extend([x0, y0, u0, v0])
            vertices.extend([x0, y1, u0, v1])
            vertices.extend([x1, y1, u1, v1])

            # Triangle 2
            vertices.extend([x0, y0, u0, v0])
            vertices.extend([x1, y1, u1, v1])
            vertices.extend([x1, y0, u1, v0])

            cursor_x += cw

        if not vertices:
            return

        vertex_data = np.array(vertices, dtype=np.float32)
        num_vertices = len(vertices) // 4  # 4 floats per vertex

        # ── Upload to VBO ─────────────────────────────────────────────────
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)

        # ── Set shader uniforms ───────────────────────────────────────────
        self._shader.use()
        self._shader.set_mat4("uProjection", self._projection)
        self._shader.set_vec3("uTextColor", color)
        self._shader.set_float("uOpacity", 1.0)
        self._shader.set_int("uFontAtlas", 0)

        # ── Bind atlas texture ────────────────────────────────────────────
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._atlas_texture)

        # ── Draw ──────────────────────────────────────────────────────────
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, num_vertices)
        glBindVertexArray(0)

    def render_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        color: Tuple[float, float, float] = (0.0, 0.8, 1.0),
        opacity: float = 0.4,
    ) -> None:
        """Render a solid semi-transparent rectangle in screen space."""
        vertices = [
            x, y, 0.5, 0.5,
            x, y + h, 0.5, 0.5,
            x + w, y + h, 0.5, 0.5,

            x, y, 0.5, 0.5,
            x + w, y + h, 0.5, 0.5,
            x + w, y, 0.5, 0.5,
        ]
        vertex_data = np.array(vertices, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)

        self._shader.use()
        self._shader.set_mat4("uProjection", self._projection)
        self._shader.set_vec3("uTextColor", color)
        self._shader.set_float("uOpacity", opacity)
        self._shader.set_int("uFontAtlas", 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._white_texture)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)
        glBindVertexArray(0)

    def render_rect_outline(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        thickness: float = 2.0,
        color: Tuple[float, float, float] = (0.0, 0.8, 1.0),
        opacity: float = 0.9,
    ) -> None:
        """Render a rectangular outline border in screen space."""
        # Top
        self.render_rect(x, y, w, thickness, color, opacity)
        # Bottom
        self.render_rect(x, y + h - thickness, w, thickness, color, opacity)
        # Left
        self.render_rect(x, y, thickness, h, color, opacity)
        # Right
        self.render_rect(x + w - thickness, y, thickness, h, color, opacity)

    def render_button(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        is_hovered: bool = False,
        color: Tuple[float, float, float] = (0.0, 0.8, 1.0),
    ) -> None:
        """Render a modern interactive UI button with hover states and text."""
        # Background fill
        bg_opacity = 0.75 if is_hovered else 0.45
        bg_color = (0.0, 0.45, 0.65) if is_hovered else (0.05, 0.15, 0.28)
        self.render_rect(x, y, w, h, color=bg_color, opacity=bg_opacity)

        # Border outline
        border_color = (0.3, 1.0, 1.0) if is_hovered else color
        border_opacity = 1.0 if is_hovered else 0.85
        self.render_rect_outline(x, y, w, h, thickness=2.0, color=border_color, opacity=border_opacity)

        # Calculate text position (centered and dynamically fitted inside box)
        scale = min(0.24, (w - 12.0) / max(1.0, len(text) * self._char_width))
        cw = self._char_width * scale
        ch = self._char_height * scale
        text_w = len(text) * cw
        text_x = x + (w - text_w) / 2.0
        text_y = y + (h - ch) / 2.0

        text_color = (1.0, 1.0, 1.0) if is_hovered else (0.0, 0.9, 1.0)
        self.render_text(text, text_x, text_y, scale=scale, color=text_color)

    # ──────────────────────────────────────────────────────────────────────────
    # State management
    # ──────────────────────────────────────────────────────────────────────────

    def begin(self) -> None:
        """Prepare GL state for HUD overlay rendering.

        Disables depth testing and sets alpha blending for text compositing.
        """
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def end(self) -> None:
        """Restore GL state after HUD rendering."""
        glEnable(GL_DEPTH_TEST)

    def resize(self, width: int, height: int) -> None:
        """Update the orthographic projection for a new screen size.

        Args:
            width: New screen width in pixels.
            height: New screen height in pixels.
        """
        self._width = width
        self._height = height
        self._projection = self._ortho_projection(width, height)
        logger.debug("HUDRenderer resized to %dx%d", width, height)

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release all OpenGL resources held by this renderer."""
        logger.info("Cleaning up HUDRenderer")
        try:
            if self._atlas_texture:
                glDeleteTextures([self._atlas_texture])
                self._atlas_texture = 0
            if hasattr(self, '_white_texture') and self._white_texture:
                glDeleteTextures([self._white_texture])
                self._white_texture = 0
        except Exception:
            logger.warning("Failed to delete HUD textures")

        try:
            if self._vbo:
                glDeleteBuffers(1, [self._vbo])
                self._vbo = 0
        except Exception:
            logger.warning("Failed to delete HUD VBO")

        try:
            if self._vao:
                glDeleteVertexArrays(1, [self._vao])
                self._vao = 0
        except Exception:
            logger.warning("Failed to delete HUD VAO")

        try:
            if self._shader:
                self._shader.delete()
                self._shader = None
        except Exception:
            logger.warning("Failed to delete HUD shader")
