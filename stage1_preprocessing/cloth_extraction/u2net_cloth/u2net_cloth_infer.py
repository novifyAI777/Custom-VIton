"""
Auto-generated template file
U2Net cloth infer template
"""

from PIL import Image
import numpy as np

class U2NetCloth:
    def __init__(self, model_path=None):
        self.model_path = model_path
        print(f"U2NetCloth initialized (model_path={model_path})")

    def segment(self, image):
        arr = np.array(image.convert('L'))
        h,w = arr.shape
        mask = np.zeros((h,w), dtype=np.float32)
        mask[h//6:h*5//6, w//6:w*5//6] = 1.0
        return mask

    def save(self, mask, out_path):
        from PIL import Image
        Image.fromarray((mask*255).astype('uint8')).save(out_path)
