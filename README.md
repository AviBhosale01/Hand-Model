# 🌌 AR Holographic Gesture-Controlled 3D Object Viewer

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3%20Core-5586A4?style=for-the-badge&logo=opengl&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-0097A7?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

<p align="center">
  <b>A real-time, Iron Man-style 3D Augmented Reality Hologram Viewer powered by MediaPipe AI and PyOpenGL.</b>
</p>

</div>

---

## 🌟 Overview

**AR Holographic Viewer** turns your webcam feed into a futuristic, interactive holographic display. Using **MediaPipe Tasks** for real-time 478-point face mesh and dual-hand tracking, the engine anchors glowing 3D objects in front of your face and lets you scale, cycle, and orient models using natural hand gestures and a 3D control panel.

The graphics engine is built from scratch with **OpenGL 3.3 Core Profile** featuring custom GLSL shaders, HDR Bloom post-processing, particle simulations, and real-time skeleton overlay drawing.

---

## ✨ Key Features

* 🕶️ **Futuristic Holographic Aesthetic**: Custom GLSL shaders with neon glow, Blinn-Phong lighting, Fresnel rim highlights, and dynamic scanning overlays.
* 👤 **478-Point Face Mesh & Head Anchor**: Projects the hologram container in front of your head using MediaPipe Face Landmarker. Features cyan face mesh dots and a red nose-anchor tracking indicator.
* 👐 **Dual-Hand Tracking Skeletons**:
  * **Left Hand (Yellow Skeleton)**: Controls container scaling (Pinch to shrink/expand, Open/Close palm).
  * **Right Hand (Green Skeleton)**: Gesture-controlled model cycling (Fist $\rightarrow$ Open Palm sequence).
* 🎛️ **3-Axis 90° Interactive Orientation Panel**:
  * On-screen buttons: `[ X-Rot 90 ]`, `[ Y-Rot 90 ]`, `[ Z-Rot 90 ]`, and `[ Reset 0 deg ]`.
  * Allows rotating 3D models 90° along Pitch, Yaw, or Roll with a single click.
  * Keyboard shortcuts: Press `X`, `Y`, `Z` keys.
* 💾 **Per-Model Isolated Rotation Memory**: Each 3D model remembers its own custom orientation angles independently when cycling between assets.
* 📷 **Glare-Free Camera Background**: Camera feed is rendered directly to the framebuffer without bloom FBO glare, preserving webcam image quality.
* 🌈 **HDR Bloom & Particle Systems**: Multi-pass separable Gaussian blur composited on top of the hologram without affecting background pixels.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| 🐍 **Core Language** | Python 3.8 – 3.13 | Multithreaded architecture & application coordination. |
| 🎮 **Graphics Engine** | OpenGL 3.3 Core | Custom GLSL shaders (Vertex, Fragment, Post-processing FBOs). |
| 🤖 **AI Tracking** | MediaPipe Tasks | 478-point Face Landmarker & Dual-Hand Landmarker tracking. |
| 📹 **Vision & Video** | OpenCV (cv2) | High-speed multithreaded camera capture and skeleton drawing. |
| 🧊 **3D Asset Engine** | Trimesh & Pyrr | GLB/GLTF/OBJ loading, unit normalization, and matrix math. |
| 🖥️ **Windowing & HUD** | GLFW & Pillow | Native window creation, VSync, and procedural font atlas text rendering. |

---

## 📁 Project Architecture

```
Antigravity/
├── main.py                     # Master Coordinator & Application Loop
├── config.yaml                 # System settings (thresholds, speeds, debug options)
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
│
├── camera/
│   └── capture.py              # Multithreaded OpenCV camera frame grabber
│
├── tracking/
│   ├── hand_tracker.py         # MediaPipe Hand Landmarker wrapper (21 joints/hand)
│   ├── face_tracker.py         # MediaPipe 478-point Face Landmarker wrapper
│   └── draw_utils.py           # Real-time skeleton & face mesh overlay renderer
│
├── gesture/
│   ├── detector.py             # Pinch, fist, and open-palm gesture detection algorithms
│   ├── debounce.py             # Time-based debouncers preventing false triggers
│   └── state_machine.py        # FSM with per-model orientation memory
│
├── graphics/
│   ├── window.py               # GLFW Window context, HiDPI scaling & mouse dispatchers
│   ├── shader.py               # Shader compiler & uniform location binder
│   └── gl_utils.py             # VAO, VBO, EBO, and FBO helper routines
│
├── renderer/
│   ├── scene.py                # Coordinates 3D scene rendering, bloom FBOs, and HUD
│   ├── background.py           # Renders live webcam background (glare-free)
│   ├── cube_renderer.py        # Holographic wireframe container renderer
│   ├── model_renderer.py       # Blinn-Phong & Fresnel 3D mesh renderer
│   ├── particle_renderer.py    # Additive energy particle sprite system
│   └── hud_renderer.py         # 2D Orthographic HUD & interactive UI button renderer
│
├── models/
│   └── loader.py               # 3D mesh loader, node hierarchy unpacker & normalizer
│
├── effects/
│   └── bloom.py                # High Dynamic Range (HDR) Bloom post-processing pipeline
│
└── shaders/                    # GLSL Shaders (330 Core Profile)
    ├── background.*            # Camera texture shaders
    ├── hologram.*              # Wireframe cube container shaders
    ├── model.*                 # Blinn-Phong model shaders
    ├── particle.*              # Additive particle shaders
    └── hud.*                   # UI Text & Button quads shaders with opacity control
```

---

## ⚡ Installation & Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AviBhosale01/Hand-Model.git
cd Hand-Model
```

### 2️⃣ Install Python Dependencies
```bash
pip install -r requirements.txt
```

> [!TIP]
> Compatible with **Python 3.8 up to Python 3.13** on Windows, macOS, and Linux.

### 3️⃣ Launch the Application
```bash
python main.py
```

> [!NOTE]
> MediaPipe AI models (`hand_landmarker.task` and `face_landmarker.task`) will automatically download to the `assets/` directory on first launch if not present.

---

## 🎮 Controls & Keyboard Shortcuts

| Control | Action | Description |
| :---: | :--- | :--- |
| 🖱️ **UI Buttons** | `X-Rot 90` / `Y-Rot 90` / `Z-Rot 90` / `Reset` | Rotate active 3D model 90° along Pitch, Yaw, Roll, or Reset. |
| ⌨️ **`X`** | Pitch 90° | Rotate model +90° vertically along X-axis. |
| ⌨️ **`Y`** | Yaw 90° | Rotate model +90° horizontally along Y-axis. |
| ⌨️ **`Z`** | Roll 90° | Rotate model +90° sideways along Z-axis. |
| ⌨️ **`D`** | Toggle HUD & Landmarks | Show/hide tracking skeletons, face mesh, and debug HUD overlay. |
| ⌨️ **`F11`** | Toggle Fullscreen | Switch between windowed mode and borderless fullscreen. |
| ⌨️ **`Ctrl + R`** | Hot-Reload Shaders | Reload all GLSL shaders in real-time without restarting. |
| ⌨️ **`S`** | Take Screenshot | Capture current frame buffer and save to `screenshots/`. |
| ⌨️ **`ESC`** | Quit Application | Safely release webcam threads and GPU resources. |

---

## 📦 Adding Custom 3D Models

1. Drop your `.glb` or `.gltf` 3D files into the **`assets/`** folder.
2. The engine automatically normalizes mesh scales to fit inside the holographic container and extracts original vertex materials and colors.
3. Make a **Right Hand Fist $\rightarrow$ Open Palm** sequence to cycle through your models in real time!

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.