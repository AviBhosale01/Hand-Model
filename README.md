# 🌌 AR Holographic Gesture-Controlled 3D Object Viewer

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3%20Core-5586A4?style=for-the-badge&logo=opengl&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20AI-0097A7?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-F5A623?style=for-the-badge&logo=open-source-initiative&logoColor=white)

<br/>

### 🔮 *Next-Gen Iron Man-style Spatial 3D Augmented Reality Interface*
**Crafted with ❤️ by Avii**

<br/>

[✨ Features](#-key-features) • [👐 Gesture Guide](#-step-by-step-hand-gesture-guide) • [🛠️ Tech Stack](#️-technology-stack) • [⚡ Installation](#-installation--quick-start) • [🎮 Keyboard Controls](#-keyboard-controls--shortcuts) • [📦 Custom Models](#-adding-custom-3d-models)

</div>

---

## 🌟 Overview

**AR Holographic Viewer** turns your standard webcam feed into a futuristic, interactive spatial computing interface. Powered by **MediaPipe Tasks AI** for real-time 478-point face mesh tracking and dual-hand skeletal joint detection, the engine anchors 3D holographic containers directly in front of your head.

The graphics engine is engineered from scratch on **OpenGL 3.3 Core Profile** with custom multi-pass GLSL shaders, texture coordinate sampling, separable Gaussian HDR Bloom post-processing, orbital particle simulations, and real-time biometric HUD overlays.

---

## ✨ Key Features

| Feature | Description | Highlight |
| :--- | :--- | :---: |
| 👑 **Signature Branding** | On-screen glassmorphic **`Made By Avii`** gold emblem in top-right HUD. | `Branding` |
| 🎨 **Solid Opaque Mode (Default)** | 3D models render with **true original textures & vibrant colors** with Blinn-Phong lighting. | `True Color` |
| 🕶️ **Hologram Neon Glow Mode** | Toggleable sci-fi cyan glow with Fresnel boundary edge-lighting & scanline passes. | `Futuristic` |
| 👤 **478-Point Face Mesh Anchor** | Head-anchored 3D viewport using MediaPipe Face Landmarker with nose-tip tracking. | `Spatial 3D` |
| 👐 **Dual-Hand Skeleton AI** | Left Hand (Yellow skeleton) for scaling & summoning; Right Hand (Green) for model cycling. | `Zero-Touch` |
| ⌨️ **High-Contrast Control HUD** | Live 3D orientation readout and yellow shortcut key hints (`[H]`, `[X]`, `[Y]`, `[Z]`, `[0]`). | `Telemetry` |
| 💾 **Per-Model Rotation Memory** | Each model independently remembers its custom pitch, yaw, and roll orientations. | `Persistence` |
| 📷 **Glare-Free Camera Pipeline** | Direct framebuffer camera feed with zero bloom glare for pristine video clarity. | `60 FPS` |

---

## 👐 Step-by-Step Hand Gesture Guide

Control your 3D assets entirely through natural hand gestures recognized in real-time by MediaPipe AI.

### 1️⃣ Right Hand: 3D Model Cycling (Fist $\rightarrow$ Open Palm Sequence)

Use your **Right Hand (Green Skeleton)** to cycle to the next 3D asset in your `assets/` directory:

```
┌─────────────────────────┐          ┌─────────────────────────┐          ┌───────────────────────────┐
│   Step 1: Make a Fist   │          │   Step 2: Open Palm     │          │         Result            │
│  ✊ Close all 5 fingers │   ───>   │  🖐️ Spread fingers wide │   ───>   │ 🔄 Cycles to Next 3D Mesh │
└─────────────────────────┘          └─────────────────────────┘          └───────────────────────────┘
```

#### Detailed Breakdown:
1. **Raise your Right Hand**: Hold your right hand within the webcam's field of view.
2. **Form a Closed Fist (✊)**: Curl all 5 fingers tightly inward. The tracker logs `is_fist = True`.
3. **Open your Palm (🖐️)**: Snap your hand open with fingers extended. The state machine transitions from **Fist $\rightarrow$ Open Palm** and cycles to the next model with an elastic entrance animation!
4. **Repeat to Cycle**: Simply repeat the sequence anytime to browse through all models in your catalog.

---

### 2️⃣ Left Hand: Hologram Container & Scaling Controls

Use your **Left Hand (Yellow Skeleton)** to summon, dismiss, and dynamically resize the 3D hologram:

* 🖐️ **Summon Hologram (Open Palm)**: Raise an open left palm to smoothly expand the 3D model and wireframe cube into view.
* ✊ **Dismiss Hologram (Closed Fist)**: Close your left hand into a fist to shrink and dismiss the container.
* 👌 **Pinch Scaling (Thumb & Index)**:
  * **Pinch Close (Fingers Touching)**: Smoothly scales down the holographic container.
  * **Pinch Open (Fingers Apart)**: Expands the container and 3D asset to full scale.

---

## 🛠️ Technology Stack

<div align="center">

```
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                               APPLICATION STACK                                  │
 ├───────────────────┬───────────────────┬───────────────────┬──────────────────────┤
 │   ⚡ Python 3.13  │   🎮 OpenGL 3.3   │  🤖 MediaPipe AI  │   📹 OpenCV 4.x      │
 │   Core Runtime    │   GPU Pipeline    │   Vision Models   │   Video Capture      │
 ├───────────────────┼───────────────────┼───────────────────┼──────────────────────┤
 │   🧊 Trimesh GLTF │   📐 Pyrr Math    │   🖥️ GLFW 3.4     │   🎨 Custom GLSL     │
 │   Asset Parser    │   Matrix Engine   │   Window & VSync  │   Shaders & Bloom    │
 └───────────────────┴───────────────────┴───────────────────┴──────────────────────┘
```

</div>

| Layer | Technologies | Role & Architecture |
| :--- | :--- | :--- |
| 🧠 **AI & Computer Vision** | MediaPipe Tasks, OpenCV, NumPy | Real-time 478-pt Face Mesh, 21-joint Hand Landmarking, One-Euro jitter reduction. |
| 🎨 **Rendering Engine** | PyOpenGL, GLSL 330 Core, GLFW | Multi-pass HDR Bloom FBOs, Blinn-Phong vertex lighting, procedural HUD atlas. |
| 🧊 **3D Geometry & Math** | Trimesh, Pyrr, Pillow | GLB/GLTF texture map sampling, vertex normalization, Euler & Quaternion transforms. |
| ⚙️ **Interaction Architecture** | Python State Machines, Debouncers | Robust FSM with per-model orientation memory and gesture cooldown filters. |

---

## 📁 System Architecture

```
Antigravity/
├── main.py                     # Master Application Coordinator & Main Loop
├── config.yaml                 # System configurations (thresholds, speeds, colors)
├── requirements.txt            # System dependencies
├── README.md                   # Documentation & Gesture Guide
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
│   ├── detector.py             # Pinch, fist, and open-palm detection algorithms
│   ├── debounce.py             # Time-based debounce filters for stable triggers
│   └── state_machine.py        # FSM with per-model isolated orientation memory
│
├── graphics/
│   ├── window.py               # GLFW Window context, HiDPI scaling & mouse dispatchers
│   ├── shader.py               # Shader compiler & uniform location binder
│   └── gl_utils.py             # VAO, VBO, EBO, and FBO allocation helpers
│
├── renderer/
│   ├── scene.py                # Coordinates 3D rendering, bloom FBOs, and HUD
│   ├── background.py           # Renders live webcam background (glare-free)
│   ├── cube_renderer.py        # Holographic wireframe container renderer
│   ├── model_renderer.py       # Solid Blinn-Phong & Holographic 3D mesh renderer
│   ├── particle_renderer.py    # Additive glowing point-sprite particle system
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
> Fully compatible with **Python 3.8 through 3.13** across Windows, macOS, and Linux.

### 3️⃣ Launch the Application
```bash
python main.py
```

> [!NOTE]
> Pre-trained MediaPipe AI models (`hand_landmarker.task` and `face_landmarker.task`) will automatically download to the `assets/` directory on first launch if not present.

---

## 🎮 Keyboard Controls & Shortcuts

| Key | Action | Functionality |
| :---: | :--- | :--- |
| ⌨️ **`H` / `M`** | **Toggle Render Style** | Switch between **Solid Original Colors Mode** (default) and **Hologram Blue Glow Mode**. |
| ⌨️ **`X`** | **Pitch 90°** | Rotate active 3D model +90° vertically along the X-axis. |
| ⌨️ **`Y`** | **Yaw 90°** | Rotate active 3D model +90° horizontally along the Y-axis. |
| ⌨️ **`Z`** | **Roll 90°** | Rotate active 3D model +90° sideways along the Z-axis. |
| ⌨️ **`0`** | **Reset Rotation** | Reset active 3D model orientation angles back to 0°. |
| ⌨️ **`D`** | **Toggle Debug HUD** | Show/hide tracking skeletons, face mesh, and telemetry overlay. |
| ⌨️ **`F11`** | **Toggle Fullscreen** | Switch between windowed mode and borderless fullscreen. |
| ⌨️ **`Ctrl + R`** | **Hot-Reload Shaders** | Live reload all GLSL shaders from disk without restarting. |
| ⌨️ **`S`** | **Take Screenshot** | Capture the active framebuffer and save directly to `screenshots/`. |
| ⌨️ **`ESC`** | **Quit Application** | Safely release webcam threads and clean up GPU memory buffers. |

---

## 📦 Adding Custom 3D Models

1. Drop your favorite **`.glb`**, **`.gltf`**, or **`.obj`** 3D model files into the **`assets/`** directory.
2. The engine automatically normalizes mesh bounding boxes and samples texture maps directly into vertex buffers.
3. Make a **Right Hand Fist $\rightarrow$ Open Palm** sequence to cycle to your newly added 3D models in real time!

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

<div align="center">
  <sub>Built with passion for Augmented Reality & Computer Vision by <b>Avii</b>.</sub>
</div>