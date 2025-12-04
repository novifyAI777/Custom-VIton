"""
Auto-generated template file
SCHP parsing template
"""

from PIL import Image
import numpy as np

class SCHPParser:
    def __init__(self, checkpoint=None, device='cpu'):
        self.checkpoint = checkpoint
        self.device = device
        print(f"SCHPParser initialized (checkpoint={checkpoint})")

    def parse(self, image):
        """Return a label map (H x W) numpy array with integer labels."""
        arr = np.array(image.convert('L'))
        h,w = arr.shape
        labels = np.zeros((h,w), dtype=np.uint8)
        # placeholder: label torso area as 1, rest 0
        labels[h//4:h//2, w//4:w*3//4] = 1
        return labels

    def save_parsing(self, labels, out_path):
        from PIL import Image
        Image.fromarray((labels*30).astype('uint8')).save(out_path)
