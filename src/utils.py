from PIL import Image
import numpy as np

def load_image(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)

def save_image(array, path):
    img = Image.fromarray(array)
    img.save(path)

def load_mask(path):
    # Load mask and convert to binary (0 and 255/1)
    mask = Image.open(path).convert('L')
    return np.array(mask)
