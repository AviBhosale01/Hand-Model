"""
Renderer package for AR Holographic Gesture Controlled 3D Object Viewer.

Exports the master SceneRenderer which orchestrates all rendering subsystems
including background camera feed, holographic cube, 3D model display,
particle effects, bloom post-processing, and HUD overlay.
"""

from renderer.scene import SceneRenderer

__all__ = ['SceneRenderer']
