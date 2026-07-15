# 🌌 AR Holographic Gesture-Controlled 3D Object Viewer

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![OpenGL Version](https://img.shields.io/badge/opengl-3.3%20Core-orange.svg)](https://www.opengl.org/)
[![MediaPipe](https://img.shields.io/badge/mediapipe-tasks-green.svg)](https://google.github.io/mediapipe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industry-grade, real-time 3D AR visualization platform that converts a standard webcam feed into a futuristic, Iron Man-style holographic interface. Using state-of-the-art **MediaPipe Tasks** for real-time face and dual-hand tracking, the application projects interactive 3D meshes inside a glowing holographic container anchored in front of the user's face.

The rendering pipeline is powered by **OpenGL 3.3 Core Profile** and features a custom multi-pass GPU-accelerated Post-Processing Bloom pipeline, real-time particle simulation, and edge-glowing Fresnel shaders.

---

## ✨ Features

*   **🕶️ Futuristic Holographic Aesthetic**: Custom GLSL shaders render objects with a neon glow, pulsing animations, Fresnel boundary highlights, and horizontal digital scanning line overlays.
*   **👤 Head-Anchored Viewport**: Tracks the user's face center and distance in real time, projecting the hologram container approximately 25cm in front of their head. Includes One-Euro jitter filters for smooth, latency-free tracking translation.
*   **👐 Dual-Hand Interaction Gestures**:
    *   **Left Hand (Container Control)**: Pinch your thumb and index finger together to shrink/enlarge the holographic cuboid. Open your fingers apart to summon (expand) the container; pinch them closed completely to vanish (collapse) it.
    *   **Right Hand (Content Control)**: Make a **closed fist** and then **open your palm** (Fist $\rightarrow$ Open Palm transition sequence) to cycle to the next 3D model.
*   **✨ Particle Simulation**: An active orbital ring of glowing energy sparks surrounds the holographic container, reacting in size and speed to the container's state.
*   **🌈 HDR Bloom Post-Processing**: Renders the holographic scene to a High Dynamic Range (HDR) Framebuffer Object (FBO), extracts bright areas, blurs them using horizontal/vertical ping-pong Gaussian filters, and composites them back onto the camera feed using Reinhard tone mapping and gamma correction.
*   **🏎️ Optimized Real-Time Engine**: Achieves a target 60 FPS by running camera capture in a separate background thread, converting frames asynchronously, and uploading vertices once to GPU memory using Vertex Array Objects (VAOs) and Element Buffer Objects (EBOs).

---

## 🛠️ Technology Stack

| Technology | Purpose | Description |
| :--- | :--- | :--- |
| **Python** | Core Language | Application architecture & coordination. |
| **OpenGL 3.3 Core** | Render Pipeline | High-performance GPU hardware rendering. |
| **MediaPipe Tasks** | Tracking | AI-powered face detection and hand landmark tracking (compatible with Python 3.12/3.13). |
| **GLFW & PyOpenGL** | Context & Windowing | GLFW window context management & OpenGL bindings. |
| **Trimesh** | 3D Asset Management | Parsing, concatenating, and normalizing 3D meshes. |
| **Pyrr** | Mathematics | Vector, matrix, and quaternion math for OpenGL transforms. |
| **Pillow (PIL)** | Font Engine | Procedural rasterization of a glyph atlas for the HUD display. |

---

## 📂 System Architecture

```
Antigravity/
├── main.py                     # Master Application Coordinator & Entry Point
├── config.yaml                 # Configuration File (thresholds, speeds, & colors)
├── requirements.txt            # System python dependencies
├── .gitignore                  # Git exclusion rules
│
├── camera/
│   └── capture.py              # Multithreaded OpenCV camera frame grabber
│
├── tracking/
│   ├── hand_tracker.py         # MediaPipe Hand Landmarker wrapper
│   ├── face_tracker.py         # MediaPipe Face Detector wrapper
│   └── smoothing.py            # One-Euro & EMA jitter-reduction filters
│
├── gesture/
│   ├── detector.py             # Pinch, fist, and open-palm detection algorithms
│   ├── debounce.py             # Time-based debounce filter to prevent false triggers
│   └── state_machine.py        # Finite State Machine (FSM) managing interaction states
│
├── graphics/
│   ├── window.py               # GLFW Window context initialization & callback loops
│   ├── shader.py               # Shader compiler & uniform uName binder
│   └── gl_utils.py             # VAO, VBO, EBO, and FBO allocation helpers
│
├── renderer/
│   ├── scene.py                # Coordinates rendering of the 3D scene & bloom
│   ├── background.py           # Renders the live webcam texture (correctly oriented)
│   ├── cube_renderer.py        # Renders the glowing holographic container
│   ├── model_renderer.py       # Fresnel edge glowing 3D mesh renderer
│   ├── particle_renderer.py    # Additive glowing point-sprite particle system
│   └── hud_renderer.py         # 2D screen-space orthographic HUD text renderer
│
├── models/
│   └── loader.py               # 3D mesh loading and unit-box normalization
│
├── animations/
│   ├── animator.py             # Tween manager for value animations
│   └── easing.py               # Ease curves (out-elastic, in-cubic, etc.)
│
├── effects/
│   └── bloom.py                # High Dynamic Range (HDR) bloom blur pipeline
│
└── shaders/                    # GLSL Vertex & Fragment Shaders (330 Core)
    ├── background.*            # Camera frame rendering and vignette shaders
    ├── hologram.*              # Holographic wireframe container shaders
    ├── model.*                 # Fresnel edge scanline model shaders
    ├── particle.*              # Additive point-sprite particle shaders
    ├── hud.*                   # Text rendering HUD shaders
    ├── passthrough.vert        # Post-processing vertex shader
    ├── bloom_extract.frag      # Brightness extraction fragment shader
    ├── bloom_blur.frag         # Separable Gaussian blur fragment shader
    └── bloom_combine.frag      # Tone-mapping screen composite shader
```

---

## 🚀 Installation & Setup

### Prerequisites

*   **Python 3.8 - 3.13** (64-bit version recommended).
*   A webcam.
*   A graphics processor (GPU) supporting **OpenGL 3.3 Core Profile** or higher.

### Step 1: Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```
> [!NOTE]
> If `PyOpenGL-accelerate` fails to compile under Windows, you can safely skip/uninstall it. The standard `PyOpenGL` library handles all bindings correctly.

### Step 2: Download Tracking Models (Offline Mode Setup)
The application works completely offline. Place the pre-trained MediaPipe AI models directly inside your **`assets/`** folder. If they are missing, the application will download them at first launch:

*   **Hand Landmarker model**: [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) (~5.6 MB)
*   **Face Detector model**: [blaze_face_short_range.tflite](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite) (~2.3 MB)

Place both files directly into:
`Antigravity/assets/`

---

## 🗃️ Adding Custom 3D Models

1. Drop your favorite 3D models into the **`assets/`** folder.
2. **Supported Formats**: 
   * **`.glb` / `.gltf`** (glTF — **recommended** for best performance and built-in normals).
   * **`.obj`** (Wavefront OBJ).
   * **`.stl`** (STereolithography).
   * **`.ply`** (Polygon File Format).
   > [!WARNING]
   > Proprietary formats like `.fbx` are not natively supported by the Python `trimesh` library and will fail to parse. Convert them to `.glb` or `.obj` using Blender or an online converter before placing them in the `assets/` folder.
3. **Auto-Normalization**: The engine automatically centers any model at $(0, 0, 0)$ and normalizes its scale to fit inside a standard $1.0 \times 1.0 \times 1.0$ unit volume, meaning you don't have to adjust mesh scales manually.

---

## ⌨️ Controls & Keyboard Shortcuts

*   **F11**: Toggle Fullscreen mode.
*   **D**: Toggle HUD Overlay (displays FPS, active FSM state, and loaded model name).
*   **R**: Hot-Reload all GLSL Shaders from disk (modify shaders in `/shaders` and load changes instantly without restarting the app!).
*   **S**: Take a screenshot (saved to `screenshots/` folder).
*   **ESC**: Gracefully stop camera thread, release GPU resources, and close the application.

---

## 🔄 Interaction FSM (Finite State Machine)

The state-handling logic is isolated inside `gesture/state_machine.py` and cycles as follows:

```
                 +--------------+
                 |     IDLE     | <------------------------------------+
                 +--------------+                                      |
                        |                                              |
                        | Left Hand Pinch Open (Thumb/Index apart)     |
                        v                                              |
             +--------------------+                                    |
             |   CUBE_APPEARING   |                                    |
             +--------------------+                                    |
                        |                                              |
                        | Elastic Grow Tween Complete                  |
                        v                                              |
             +--------------------+                                    | Left Pinch Close
             |    CUBE_ACTIVE     | -----------------------------------+ (Thumb/Index touch)
             +--------------------+                                    |
               |                |                                      |
     Right Fist|->Open          | Left Pinch Close                     |
               v                v                                      |
      +-----------------+   +---------------------+                    |
      |  MODEL_CYCLING  |   |   CUBE_SHRINKING    | -------------------+
      +-----------------+   +---------------------+
               |                      |
               | Transition complete  | Shrink Tween Complete
               v                      v
         Back to ACTIVE          Back to IDLE
```

---

## ⚡ Performance Optimization

If you experience frame rate drops or lag:
1. **Downscale Resolution**: Open `config.yaml` and reduce `camera.width` / `camera.height` (e.g., to `640x480`). This reduces camera frame processing overhead and texture upload bandwidth.
2. **Reduce Blur Passes**: Set `visual.bloom_blur_passes` in `config.yaml` to `2` or `3` instead of `5`.
3. **Optimize Mesh Poly-Count**: Avoid loading meshes with millions of triangles. Use low-poly decimated models for the smoothest rendering performance.
