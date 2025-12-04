"""
Auto-generated template file
Image utility helpers
"""

from PIL import Image

def load_image(path):
    return Image.open(path).convert('RGB')

def save_image(img, path):
    img.save(path)
