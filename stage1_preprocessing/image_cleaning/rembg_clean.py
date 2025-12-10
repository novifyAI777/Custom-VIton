"""
Stage 1 - Image Cleaning using rembg (Background Removal)
"""

from rembg import remove 
from PIL import Image
import numpy as np
import os


def clean_background(input_path, output_path, mask_output_path=None):
    """
    Removes background using rembg and saves:
    - A clean PNG with transparent background
    - Optional binary mask (white = 1, black = 0)
    """
    img = Image.open(input_path).convert("RGB")

    # run rembg inference
    result = remove(img)

    # Save person clean image
    result.save(output_path)

    # Generate mask (if required)
    if mask_output_path:
        alpha = result.split()[-1]  # alpha channel
        mask = (np.array(alpha) > 128).astype('uint8') * 255
        Image.fromarray(mask).save(mask_output_path)

    print(f"[rembg] Cleaned image saved → {output_path}")
    if mask_output_path:
        print(f"[rembg] Mask saved → {mask_output_path}")
