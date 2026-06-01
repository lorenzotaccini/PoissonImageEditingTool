# Poisson Image Editing Tool

This tool implements the techniques described in the paper "Poisson Image Editing" by Pérez et al. (2003). It allows for seamless cloning, mixed gradients, and other gradient-domain image manipulations.

## Requirements
- Python 3.9+
- NumPy
- SciPy
- Pillow
- Scikit-Image

## Installation
```bash
pip install numpy scipy pillow scikit-image
```

## Usage
### Advanced Interactive UI (Recommended)
Launch the professional all-in-one editor:
```bash
python src/main.py
```
1. **Load Destination:** Choose your background image.
2. **Load Source:** Choose the image to clone from.
3. **Select Region:** Draw a polygon on the source image and press **ENTER**.
4. **Position & Scale:** Use the mouse to drag the selection on the destination. Use the **Scale Slider** to resize it.
5. **Choose Mode:** Select from Seamless Cloning, Mixed Gradients, Texture Flattening, or Illumination Change.
6. **Adjust Parameters:** Fine-tune edge thresholds or illumination factors.
7. **Process:** Hit the green **PROCESS** button.
8. **Save:** Export your masterpiece.

## Features
- **All-in-One Workspace:** No more switching between windows.
- **Interactive Transformation:** Drag and scale your selection visually before blending.
- **Advanced Paper Features:**
    - **Seamless Cloning & Mixed Gradients** (Section 3).
    - **Texture Flattening** (Section 4): Create flat, artistic looks while preserving edges.
    - **Local Illumination Change** (Section 4): Correct exposure or highlights seamlessly.
- **High Performance:** Hybrid solver (Direct/Iterative) with vectorized gradient calculations.
- **No OpenCV:** Built with modern Python libraries (PySide6, NumPy, SciPy, Scikit-Image).

