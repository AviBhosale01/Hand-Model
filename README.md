# 🌌 AR Holographic Gesture-Controlled 3D Object Viewer

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3%20Core-5586A4?style=for-the-badge&logo=opengl&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks-0097A7?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

<p align="center">
  <b>A real-time, Iron Man-style 3D Augmented Reality Hologram & Model Viewer powered by MediaPipe AI and PyOpenGL.</b>
</p>

</div>

---

## 🌟 Overview

**AR Holographic Viewer** turns your webcam feed into a futuristic, interactive 3D display. Using **MediaPipe Tasks** for real-time 478-point face mesh and dual-hand tracking, the engine anchors 3D models in front of your face and lets you scale, cycle, and re-orient assets using natural hand gestures and keyboard controls.

The graphics engine is built from scratch with **OpenGL 3.3 Core Profile** featuring custom GLSL shaders, texture map sampling, HDR Bloom post-processing, particle simulations, and real-time skeleton overlay drawing.

---

## ✨ Key Features

* 🎨 **Solid Opaque Original Colors (Default Mode)**: 3D models render in their **true, vibrant original textures and material colors** with realistic Blinn-Phong directional lighting and zero bloom distortion.
* 🕶️ **Futuristic Hologram Glow Mode**: Press **`H`** anytime to toggle to the neon cyan holographic aesthetic with Fresnel rim highlights and floating scanlines.
* 👤 **478-Point Face Mesh & Head Anchor**: Projects the hologram container in front of your head using MediaPipe Face Landmarker. Features cyan face mesh dots and a red nose-anchor tracking indicator.
* 👐 **Dual-Hand Tracking Skeletons**:
  * **Left Hand (Yellow Skeleton)**: Controls container scaling (Pinch to shrink/expand, Open/Close palm).
  * **Right Hand (Green Skeleton)**: Gesture-controlled model cycling (Fist $\rightarrow$ Open Palm sequence).
* ⌨️ **On-Screen Keyboard Shortcut Panel**:
  * Clean, high-contrast HUD panel displaying live 3D orientation angles, current render style, and bright yellow keyboard hints (`Press H`, `Press X`, `Press Y`, `Press Z`, `Press 0`).
* 💾 **Per-Model Isolated Rotation Memory**: Each 3D model maintains its own custom orientation angles independently when cycling between assets.
* 📷 **Glare-Free Camera Background**: Camera feed is rendered directly to the framebuffer without bloom FBO glare, preserving crystal-clear webcam quality.
* 🌈 **HDR Bloom & Particle Systems**: Multi-pass separable Gaussian blur composited over holographic mode without affecting background pixels.

---

## 🛠️ Technology Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| 🐍 **Core Language** | Python 3.8 – 3.13 | Multithreaded architecture & application coordination. |
| 🎮 **Graphics Engine** | OpenGL 3.3 Core | Custom GLSL shaders (Vertex, Fragment, Post-processing FBOs). |
| 🤖 **AI Tracking** | MediaPipe Tasks | 478-point Face Landmarker & Dual-Hand Landmarker tracking. |
| 📹 **Vision & Video** | OpenCV (cv2) | High-speed multithreaded camera capture and skeleton drawing. |
| 🧊 **3D Asset Engine** | Trimesh & Pyrr | GLB/GLTF/OBJ loading, texture sampling, unit normalization, and matrix math. |
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
│   ├── window.py               # GLFW Window context, HiDPI scaling & input dispatchers
│   ├── shader.py               # Shader compiler & uniform location binder
│   └── gl_utils.py             # VAO, VBO, EBO, and FBO helper routines
│
├── renderer/
│   ├── scene.py                # Coordinates 3D scene rendering, bloom FBOs, and HUD
│   ├── background.py           # Renders live webcam background (glare-free)
│   ├── cube_renderer.py        # Holographic wireframe container renderer
│   ├── model_renderer.py       # Solid Blinn-Phong & Holographic 3D mesh renderer
│   ├── particle_renderer.py    # Additive energy particle sprite system
│   └── hud_renderer.py         # 2D Orthographic HUD text & backdrop renderer
│
├── models/
│   └── loader.py               # 3D mesh loader, texture color sampler & normalizer
│
├── effects/
│   └── bloom.py                # High Dynamic Range (HDR) Bloom post-processing pipeline
│
└── shaders/                    # GLSL Shaders (330 Core Profile)
    ├── background.*            # Camera texture shaders
    ├── hologram.*              # Wireframe cube container shaders
    ├── model.*                 # Solid & Holographic model shaders
    ├── particle.*              # Additive particle shaders
    └── hud.*                   # UI Text & Backdrop shaders with opacity control
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

## 🎮 Keyboard Controls & Shortcuts

| Hotkey | Action | Description |
| :---: | :--- | :--- |
| ⌨️ **`H` / `M`** | Toggle Render Style | Switch between **Solid Original Colors Mode** (default) and **Hologram Blue Glow Mode**. |
| ⌨️ **`X`** | Pitch 90° | Rotate active 3D model +90° vertically along X-axis. |
| ⌨️ **`Y`** | Yaw 90° | Rotate active 3D model +90° horizontally along Y-axis. |
| ⌨️ **`Z`** | Roll 90° | Rotate active 3D model +90° sideways along Z-axis. |
| ⌨️ **`0`** | Reset Rotation | Reset 3D model manual orientation back to 0°. |
| ⌨️ **`D`** | Toggle Debug HUD | Show/hide tracking skeletons, face mesh, and debug stats overlay. |
| ⌨️ **`F11`** | Toggle Fullscreen | Switch between windowed mode and borderless fullscreen. |
| ⌨️ **`Ctrl + R`** | Hot-Reload Shaders | Reload all GLSL shaders in real-time without restarting the app. |
| ⌨️ **`S`** | Take Screenshot | Capture current frame buffer and save to `screenshots/`. |
| ⌨️ **`ESC`** | Quit Application | Safely release webcam threads and GPU resources. |

---

## 📦 Adding Custom 3D Models

1. Drop your `.glb` or `.gltf` 3D files into the **`assets/`** folder.
2. The engine automatically normalizes mesh scales and bakes texture maps directly into vertex buffers.
3. Make a **Right Hand Fist $\rightarrow$ Open Palm** sequence to cycle through your models in real time!

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.