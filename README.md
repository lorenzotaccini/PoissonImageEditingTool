# Poisson Image Editing Pro

A desktop application for gradient-domain image editing, based on the research paper **"Poisson Image Editing"** (Pérez et al., SIGGRAPH 2003).

This tool provides an interface for seamless cloning and local image modifications without the need for OpenCV, utilizing optimized mathematical solvers for high-quality results.

## Key Features

### 1. Seamless Cloning (Section 3)
*   **Seamless Cloning:** Paste objects from one image to another with perfect color and lighting integration.
*   **Mixed Gradients:** Combine source and destination textures. Ideal for transparent objects or preserving background patterns (e.g., skin pores, fabric).

### 2. Selection Editing (Section 4)
*   **Texture Flattening:** "Iron out" internal textures while keeping sharp outlines. Perfect for artistic effects or smoothing surfaces.
*   **Local Illumination Change:** Correct exposure, bring out shadow details, or compress specular highlights (glare) seamlessly.
*   **Color Change (Tinting):** Redefined as independent RGB gradient scaling. Change the color of objects while preserving all original shading, highlights, and 3D volume.

### 3. Interactive UI
*   **All-in-One Workspace:** A modern PySide6 interface that manages the entire workflow.
*   **Interactive Transform:** Drag selections with the mouse and scale them (0.1x to 3.0x) using a slider or the **mouse wheel**.
*   **In-Place Editing:** A dedicated mode to select a region directly on your destination image and apply Section 4 modifications immediately.
*   **Multi-Cloning:** Support for consecutive edits. Process one object, and immediately load another to build complex compositions.
*   **Workflow Memory:** Remembers your last used directory for faster file selection.

## Requirements
- Python 3.9+
- NumPy & SciPy
- Pillow & Scikit-Image
- PySide6 (Qt6)

## Installation
```bash
pip install numpy scipy pillow scikit-image PySide6
```

## Usage

### Launching the Application
```bash
python src/main.py
```

### Workflow Steps
1.  **Load Destination:** Load your primary background image.
2.  **Selection:** 
    *   Click **"Import External Source"** to cut an object from another image.
    *   Click **"Select on Destination"** for local edits (Flattening, Illumination, Color).
3.  **Draw Mask:** Click to draw a polygon. Press **ENTER** (or click the button) to confirm.
4.  **Transform:** (External sources only) Drag the cutout to position it. Use the mouse wheel or slider to scale it.
5.  **Edit Mode:** Choose the Poisson trasformation you want to apply and adjust the contextual parameters (Thresholds, Alpha/Beta, or RGB scales).
6.  **Process:** Click the green **PROCESS** button. The solver uses a memory-efficient iterative CG backend for high-res images.
7.  **Save:** Export the final result to PNG, JPG, or BMP.

## Technical Details
*   **Hybrid Solver:** Automatically switches between `spsolve` (Direct) for speed on small regions and `cg` (Iterative) for memory efficiency on large selections.
*   **Mathematical Rigor:** Correctly implements Section 2 boundary conditions ($| N_p |$) for accurate results even when selections touch image edges.
*   **Vectorization & Parallelism:** Core gradient and Laplacian calculations are fully vectorized. Red, Green, and Blue channels are solved in **parallel** using multi-threading to maximize CPU utilization.
*   **No OpenCV:** Strictly adheres to project constraints by utilizing only the approved scientific computing stack.


