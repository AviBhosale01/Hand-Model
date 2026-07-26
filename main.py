import os
import sys
import time
import logging
import numpy as np
import glfw
from OpenGL.GL import *

from config import Settings, load_settings
from utils import setup_logging, FrameTimer, screen_to_ndc
from camera import CameraCapture
from tracking import HandTracker, FaceTracker
from gesture import GestureDetector, AppState, StateMachine
from models import ModelLoader, MeshData
from animations import AnimationManager
from graphics import Window
from renderer import SceneRenderer

logger = logging.getLogger("ARHologram.Main")

class ARHologramApp:
    """Master application class coordinating window lifecycle, webcam capture,
    tracking detectors, finite state machine updates, and rendering.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.settings = load_settings(config_path)
        global logger
        logger = setup_logging(self.settings.log_level, self.settings.log_file)
        logger.info("Initializing AR Holographic Object Viewer App...")

        # Setup core window & context
        self.window = Window(self.settings)
        self.timer = FrameTimer()
        
        # Setup background services
        self.camera = CameraCapture(self.settings)
        self.hand_tracker = HandTracker(self.settings)
        self.face_tracker = FaceTracker(self.settings)
        
        # Setup gesture and state machine
        self.gesture_detector = GestureDetector(self.settings)
        self.anim_manager = AnimationManager()
        self.state_machine = StateMachine(self.settings, self.anim_manager)
        
        # Setup models asset management
        self.model_loader = ModelLoader(self.settings)
        self.loaded_models = []
        self._current_loaded_index = -1
        
        # Setup renderer
        self.scene_renderer = SceneRenderer(self.settings)
        
        self._setup_input_callbacks()

    def _setup_input_callbacks(self) -> None:
        """Binds window callbacks to key and mouse events."""
        self.window.add_key_callback(self._on_key)
        self.window.add_mouse_button_callback(self._on_mouse_button)

    def _on_key(self, window_handle, key, scancode, action, mods) -> None:
        """Callback for keyboard events inside GLFW window."""
        if action != glfw.PRESS:
            return
            
        if key == glfw.KEY_ESCAPE:
            logger.info("ESC pressed. Initiating shutdown...")
            glfw.set_window_should_close(window_handle, True)
            
        elif key == glfw.KEY_D:
            logger.info("Toggling HUD overlay display...")
            self.scene_renderer.toggle_debug()
            self.settings.show_landmarks = not self.settings.show_landmarks
            
        elif key == glfw.KEY_F11:
            logger.info("Toggling fullscreen display...")
            self.window.toggle_fullscreen()
            # Update viewport and size inside renderer
            self.scene_renderer.resize(self.window.width, self.window.height)
            
        elif key == glfw.KEY_S:
            self._take_screenshot()
            
        elif key == glfw.KEY_O:
            logger.info("Hotkey 'O' pressed: Rotating 3D model 90°...")
            self.state_machine.rotate_model_manual(delta_y=90.0)

        elif key == glfw.KEY_R:
            # Hot reload all shaders
            logger.info("Reloading shaders...")
            try:
                self.scene_renderer.cleanup()
                self.scene_renderer.initialize()
                # Restore current model mesh if loaded
                self._update_rendered_mesh()
                logger.info("Shaders reloaded successfully!")
            except Exception as e:
                logger.error(f"Shader reload failed: {e}", exc_info=True)

    def _on_mouse_button(self, window_handle, button, action, mods) -> None:
        """Callback for mouse click events inside GLFW window."""
        if action != glfw.PRESS:
            return

        fb_x, fb_y = self.window.get_framebuffer_mouse_pos()
        btn_w, btn_h = 175.0, 44.0
        btn_x = float(self.window.width) - btn_w - 20.0
        btn_y = float(self.window.height) - btn_h - 20.0

        if btn_x <= fb_x <= btn_x + btn_w and btn_y <= fb_y <= btn_y + btn_h:
            if button == glfw.MOUSE_BUTTON_LEFT:
                if mods & glfw.MOD_SHIFT:
                    self.state_machine.rotate_model_manual(delta_y=0.0, delta_x=90.0)
                else:
                    self.state_machine.rotate_model_manual(delta_y=90.0, delta_x=0.0)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.state_machine.rotate_model_manual(delta_y=0.0, delta_x=90.0)

    def _take_screenshot(self) -> None:
        """Captures the current frame buffer and saves it to a PNG."""
        logger.info("Taking screenshot...")
        os.makedirs("screenshots", exist_ok=True)
        filename = f"screenshots/screenshot_{int(time.time())}.png"
        
        width, height = self.window.width, self.window.height
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        from PIL import Image
        image = Image.frombytes("RGB", (width, height), pixels)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image.save(filename)
        logger.info(f"Screenshot saved to {filename}")

    def run(self) -> None:
        """Main application lifecycle loop."""
        # 1. Start camera background capture thread
        if not self.camera.start():
            logger.critical("Cannot start camera. Exiting...")
            return

        # 2. Setup OpenGL graphics, compile shaders, prepare VBOs
        logger.info("Initializing OpenGL Scene Renderer...")
        self.scene_renderer.initialize()

        # 3. Scan for 3D model files in assets folder
        logger.info("Scanning for 3D assets...")
        model_paths = self.model_loader.scan_directory()
        
        # Load the default mesh first as fallback
        fallback_mesh = self.model_loader.load_default_model()
        self.scene_renderer.set_mesh(fallback_mesh)
        self.state_machine.context.current_model_name = fallback_mesh.name
        
        # Load initial custom model if available
        if model_paths:
            logger.info("Pre-loading first asset...")
            mesh = self.model_loader.load_model(0)
            if mesh:
                self.scene_renderer.set_mesh(mesh)
                self._current_loaded_index = 0
                self.state_machine.context.current_model_name = mesh.name
                
        # Register window resize callback linkage
        glfw.set_framebuffer_size_callback(self.window.handle, self._on_resize)
        # Call initial resize to conform viewport
        self.scene_renderer.resize(self.window.width, self.window.height)

        logger.info("Entering main application render loop...")
        
        # Loop until window is closed
        while not self.window.should_close():
            # Tick timer
            dt = self.timer.tick()
            
            # Poll GLFW events (key input, window movements)
            self.window.poll_events()
            
            # Retrieve latest frame from background thread
            frame_bgr = self.camera.get_frame()
            frame_rgb = self.camera.get_frame_rgb()
            
            if frame_rgb is None or frame_bgr is None or frame_rgb.size == 0 or frame_bgr.size == 0:
                # Wait for camera feed to initialize
                time.sleep(0.005)
                continue
                
            # Perform hands and face detection on the current camera frame
            hands = self.hand_tracker.process(frame_rgb)
            face = self.face_tracker.process(frame_rgb)
            
            # Distinguish left/right hands
            left_hand = None
            right_hand = None
            for hand in hands:
                # MediaPipe returns left/right labels relative to camera viewpoint.
                # In selfie/mirrored view, what feels like left hand to user is "Right" in camera coordinates.
                # We align the gesture control accordingly.
                if hand.handedness == "Right":
                    left_hand = hand  # Mirrors user's left hand
                elif hand.handedness == "Left":
                    right_hand = hand # Mirrors user's right hand

            # Face center detection used to anchor the cube
            head_ndc = None
            if face is not None:
                # Map face pixel coordinate center to normalized device coordinates (NDC, [-1, 1])
                # We project it in 3D world space inside the renderer
                head_ndc = face.center
            
            # Record model index before updates to detect cycling changes in callbacks
            prev_model_idx = self.state_machine.context.model_index

            # Update Tweens/Animations
            self.anim_manager.update(dt)
            
            # Update Finite State Machine
            self.state_machine.update(
                dt=dt,
                left_hand=left_hand,
                right_hand=right_hand,
                head_pos=head_ndc,
                gesture_detector=self.gesture_detector,
                model_count=max(1, self.model_loader.model_count) # Fallback to icosphere count 1
            )
            
            # If the state machine cycled the model index, trigger background mesh update
            if self.state_machine.context.model_index != prev_model_idx:
                self._update_rendered_mesh()

            # Draw tracking overlays if enabled
            if self.settings.show_landmarks:
                from tracking.draw_utils import draw_landmarks
                frame_to_render = frame_rgb.copy()
                draw_landmarks(frame_to_render, hands, face)
            else:
                frame_to_render = frame_rgb

            # Render scene
            self.scene_renderer.render(
                frame_rgb=frame_to_render,
                state=self.state_machine.state,
                context=self.state_machine.context,
                timer=self.timer,
                mouse_pos=self.window.get_framebuffer_mouse_pos()
            )
            
            # Swap front and back OpenGL rendering buffers
            self.window.swap_buffers()

        self.cleanup()

    def _update_rendered_mesh(self) -> None:
        """Loads and binds the active 3D model mesh based on current FSM index."""
        idx = self.state_machine.context.model_index
        count = self.model_loader.model_count
        
        if count <= 0 or idx < 0:
            # Revert to procedural shape fallback
            fallback_mesh = self.model_loader.load_default_model()
            self.scene_renderer.set_mesh(fallback_mesh)
            self.state_machine.context.current_model_name = fallback_mesh.name
            return

        # Load mesh from assets folder
        mesh = self.model_loader.load_model(idx)
        if mesh:
            self.scene_renderer.set_mesh(mesh)
            self.state_machine.context.current_model_name = mesh.name
            logger.info(f"Active display mesh updated: {mesh.name}")

    def _on_resize(self, window_handle, width, height) -> None:
        """Handle window resizing from GLFW."""
        glViewport(0, 0, width, height)
        self.scene_renderer.resize(width, height)

    def cleanup(self) -> None:
        """Release trackers, camera threads, shaders, context windows."""
        logger.info("Cleaning up application resources...")
        self.camera.release()
        self.hand_tracker.release()
        self.face_tracker.release()
        self.scene_renderer.cleanup()
        self.window.destroy()
        logger.info("App shutdown complete.")


if __name__ == "__main__":
    # Test shaders/model utility if command line arguments specify
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-shaders":
            # Command line test mode: Initialize context and build shaders to verify GLSL compliance
            print("Shader compile compliance test mode activated.")
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, False)
            temp_window = glfw.create_window(100, 100, "ShaderTest", None, None)
            glfw.make_context_current(temp_window)
            
            # Compile shaders and exit
            try:
                from graphics.shader import ShaderProgram
                b = ShaderProgram('shaders/background.vert', 'shaders/background.frag')
                h = ShaderProgram('shaders/hologram.vert', 'shaders/hologram.frag')
                m = ShaderProgram('shaders/model.vert', 'shaders/model.frag')
                p = ShaderProgram('shaders/particle.vert', 'shaders/particle.frag')
                he = ShaderProgram('shaders/passthrough.vert', 'shaders/hud.frag') # HUD fragment
                print("SUCCESS: All core shaders compiled successfully!")
                sys.exit(0)
            except Exception as e:
                print(f"FAILED: Shader compilation failed: {e}")
                sys.exit(1)

        elif sys.argv[1] == "--test-models":
            # Load meshes test
            print("Model loader verification test mode activated.")
            settings = load_settings("config.yaml")
            loader = ModelLoader(settings)
            paths = loader.scan_directory()
            if not paths:
                print("No model assets found to test. Test OK (empty assets fallback active).")
                sys.exit(0)
            try:
                mesh = loader.load_model(0)
                if mesh:
                    print(f"SUCCESS: Loaded mesh: {mesh.name}")
                    sys.exit(0)
                else:
                    print("FAILED: Model loader returned None")
                    sys.exit(1)
            except Exception as e:
                print(f"FAILED: Model loading failed: {e}")
                sys.exit(1)

    # Standard run
    app = ARHologramApp("config.yaml")
    app.run()
