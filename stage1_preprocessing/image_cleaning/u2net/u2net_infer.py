"""
Auto-generated template file
U2Net predictor template
Place model weights in the models/ folder and implement loading as needed.
"""

from PIL import Image
import numpy as np
import os

class U2NetPredictor:
    def __init__(self, model_path=None, device='cpu'):
        self.model_path = model_path
        self.device = device
        # NOTE: load your model here if available
        print(f"U2NetPredictor initialized (model_path={model_path})")

    def predict(self, image):
        """Return a binary mask (0..1) numpy array same HxW as image."""
        arr = np.array(image.convert('L')) / 255.0
        h,w = arr.shape
        mask = np.zeros((h,w), dtype=np.float32)
        hh, ww = h//4, w//4
        mask[hh:hh*3, ww:ww*3] = 1.0
        return mask

    def save_mask(self, mask, out_path):
        mask_img = (mask*255).astype('uint8')
        Image.fromarray(mask_img).save(out_path)
