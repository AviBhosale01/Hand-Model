"""Graphics package for AR Holographic Gesture Controlled 3D Object Viewer.

Provides OpenGL rendering infrastructure including shader management,
window management, and GL utility functions.
"""

from graphics.shader import ShaderProgram
from graphics.window import Window
from graphics import gl_utils

__all__ = ['Window', 'ShaderProgram', 'gl_utils']
