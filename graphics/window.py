"""GLFW window management for OpenGL 3.3 Core Profile rendering.

Provides the Window class that handles GLFW initialization, window creation,
input callbacks, fullscreen toggling, and clean shutdown.
"""

import logging
import platform
from typing import Any, Callable, List, Optional

import glfw
from OpenGL.GL import glViewport

logger = logging.getLogger(__name__)


class Window:
    """Manages a GLFW window with OpenGL 3.3 Core Profile context.

    Handles window creation, resize events, keyboard input dispatch,
    and fullscreen toggling with position/size restoration.

    Attributes:
        width: Current framebuffer width in pixels.
        height: Current framebuffer height in pixels.
        aspect_ratio: Current width/height ratio.
        handle: The underlying GLFW window handle.
    """

    def __init__(self, settings: Any) -> None:
        """Initialize GLFW and create a window.

        Args:
            settings: Configuration object with attributes:
                - window_width (int): Initial window width in pixels.
                - window_height (int): Initial window height in pixels.
                - window_title (str): Window title bar text.
                - vsync (bool): Enable vertical sync.
                - fullscreen (bool): Start in fullscreen mode.

        Raises:
            RuntimeError: If GLFW initialization or window creation fails.
        """
        self._width: int = settings.window_width
        self._height: int = settings.window_height
        self._title: str = settings.window_title
        self._key_callbacks: List[Callable] = []
        self._is_fullscreen: bool = False
        self._handle: Optional[Any] = None

        # Saved windowed-mode geometry for fullscreen toggle restoration
        self._windowed_pos_x: int = 0
        self._windowed_pos_y: int = 0
        self._windowed_width: int = self._width
        self._windowed_height: int = self._height

        # Initialize GLFW
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        logger.info("GLFW initialized successfully (version %s)", glfw.get_version_string())

        # Set OpenGL 3.3 Core Profile window hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # Required on macOS for core profile contexts
        if platform.system() == "Darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

        # Anti-aliasing
        glfw.window_hint(glfw.SAMPLES, 4)

        # Create the window
        monitor = None
        if settings.window_fullscreen:
            monitor = glfw.get_primary_monitor()
            if monitor:
                video_mode = glfw.get_video_mode(monitor)
                self._width = video_mode.size.width
                self._height = video_mode.size.height
                self._is_fullscreen = True

        self._handle = glfw.create_window(
            self._width, self._height, self._title, monitor, None
        )
        if not self._handle:
            glfw.terminate()
            raise RuntimeError(
                f"Failed to create GLFW window ({self._width}x{self._height})"
            )

        # Make context current and configure
        glfw.make_context_current(self._handle)

        # VSync
        glfw.swap_interval(1 if settings.window_vsync else 0)
        logger.info("VSync %s", "enabled" if settings.window_vsync else "disabled")

        # Query actual framebuffer size (may differ from window size on HiDPI)
        fb_width, fb_height = glfw.get_framebuffer_size(self._handle)
        self._width = fb_width
        self._height = fb_height
        glViewport(0, 0, self._width, self._height)

        # Store initial windowed position for fullscreen toggle
        if not self._is_fullscreen:
            pos_x, pos_y = glfw.get_window_pos(self._handle)
            self._windowed_pos_x = pos_x
            self._windowed_pos_y = pos_y

        # Register GLFW callbacks
        glfw.set_framebuffer_size_callback(self._handle, self._framebuffer_size_callback)
        glfw.set_key_callback(self._handle, self._key_callback_dispatcher)

        logger.info(
            "Window created: %dx%d, fullscreen=%s",
            self._width,
            self._height,
            self._is_fullscreen,
        )

    def _framebuffer_size_callback(
        self, window: Any, width: int, height: int
    ) -> None:
        """Handle framebuffer resize events.

        Args:
            window: The GLFW window that was resized.
            width: New framebuffer width in pixels.
            height: New framebuffer height in pixels.
        """
        self._width = width
        self._height = height
        glViewport(0, 0, width, height)
        logger.debug("Framebuffer resized to %dx%d", width, height)

    def _key_callback_dispatcher(
        self, window: Any, key: int, scancode: int, action: int, mods: int
    ) -> None:
        """Dispatch key events to all registered key callbacks.

        Args:
            window: The GLFW window that received the event.
            key: The GLFW key code.
            scancode: The platform-specific scancode.
            action: GLFW_PRESS, GLFW_RELEASE, or GLFW_REPEAT.
            mods: Modifier key flags.
        """
        for callback in self._key_callbacks:
            try:
                callback(window, key, scancode, action, mods)
            except Exception:
                logger.exception("Error in key callback %r", callback)

    def should_close(self) -> bool:
        """Check whether the window has been requested to close.

        Returns:
            True if the window close flag is set.
        """
        return bool(glfw.window_should_close(self._handle))

    def swap_buffers(self) -> None:
        """Swap the front and back framebuffers."""
        glfw.swap_buffers(self._handle)

    def poll_events(self) -> None:
        """Process pending window events (input, resize, etc.)."""
        glfw.poll_events()

    def destroy(self) -> None:
        """Destroy the window and terminate GLFW.

        Safe to call multiple times.
        """
        if self._handle is not None:
            logger.info("Destroying window and terminating GLFW")
            glfw.destroy_window(self._handle)
            self._handle = None
        glfw.terminate()

    def toggle_fullscreen(self) -> None:
        """Toggle between windowed and borderless fullscreen modes.

        When switching to fullscreen, the current window position and size
        are saved so they can be restored when switching back to windowed mode.
        """
        if self._handle is None:
            return

        monitor = glfw.get_primary_monitor()
        if monitor is None:
            logger.warning("No primary monitor found; cannot toggle fullscreen")
            return

        if self._is_fullscreen:
            # Restore to windowed mode
            glfw.set_window_monitor(
                self._handle,
                None,  # No monitor = windowed
                self._windowed_pos_x,
                self._windowed_pos_y,
                self._windowed_width,
                self._windowed_height,
                glfw.DONT_CARE,
            )
            self._is_fullscreen = False
            logger.info(
                "Switched to windowed mode (%dx%d at %d,%d)",
                self._windowed_width,
                self._windowed_height,
                self._windowed_pos_x,
                self._windowed_pos_y,
            )
        else:
            # Save current windowed geometry
            self._windowed_pos_x, self._windowed_pos_y = glfw.get_window_pos(
                self._handle
            )
            self._windowed_width, self._windowed_height = glfw.get_window_size(
                self._handle
            )

            # Switch to fullscreen
            video_mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self._handle,
                monitor,
                0,
                0,
                video_mode.size.width,
                video_mode.size.height,
                video_mode.refresh_rate,
            )
            self._is_fullscreen = True
            logger.info(
                "Switched to fullscreen (%dx%d @ %dHz)",
                video_mode.size.width,
                video_mode.size.height,
                video_mode.refresh_rate,
            )

    def add_key_callback(self, callback: Callable) -> None:
        """Register a key event callback.

        The callback signature must be:
            callback(window, key: int, scancode: int, action: int, mods: int)

        Args:
            callback: The callback function to register.
        """
        if callback not in self._key_callbacks:
            self._key_callbacks.append(callback)
            logger.debug("Added key callback: %r", callback)

    @property
    def width(self) -> int:
        """Current framebuffer width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Current framebuffer height in pixels."""
        return self._height

    @property
    def aspect_ratio(self) -> float:
        """Current width-to-height ratio. Returns 1.0 if height is zero."""
        if self._height == 0:
            return 1.0
        return self._width / self._height

    @property
    def handle(self) -> Any:
        """The underlying GLFW window handle."""
        return self._handle

    def __repr__(self) -> str:
        return (
            f"Window({self._width}x{self._height}, "
            f"fullscreen={self._is_fullscreen})"
        )
