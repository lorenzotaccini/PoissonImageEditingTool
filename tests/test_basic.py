import sys
import os
import numpy as np
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from solver import poisson_edit
from utils import save_image

def create_dummy_data():
    # Create a destination image (blue background)
    dest = np.zeros((200, 200, 3), dtype=np.uint8)
    dest[:, :, 2] = 200  # Blue
    
    # Create a source image (red circle on white background)
    source = np.ones((100, 100, 3), dtype=np.uint8) * 255
    yy, xx = np.ogrid[:100, :100]
    mask_circle = (xx - 50)**2 + (yy - 50)**2 < 30**2
    source[mask_circle] = [255, 0, 0] # Red circle
    
    # Create mask for the circle
    mask = (mask_circle * 255).astype(np.uint8)
    
    return source, dest, mask

def test_cloning():
    source, dest, mask = create_dummy_data()
    print("Testing seamless cloning...")
    # Paste the red circle at (50, 50) in the blue background
    result = poisson_edit(source, dest, mask, offset=(50, 50))
    save_image(result, "test_output.png")
    print("Result saved to test_output.png")

if __name__ == "__main__":
    test_cloning()
