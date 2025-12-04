# cloth_extraction/segformer_cloth.py

import os
import torch
import numpy as np
from PIL import Image

try:
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")


class SegformerClothExtractor:
    def __init__(self, model_name="sayeed99/segformer-b3-fashion", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SegFormer model: {model_name}")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("✓ Model loaded successfully")

    def extract(self, img_path, cloth_out_path, mask_out_path):
        img = Image.open(img_path).convert("RGB")
        img_size = img.size
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (1, num_classes, H, W)
            seg = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

        # convert segmentation map to mask: cloth regions >0
        mask = (seg != 0).astype(np.uint8) * 255  # background class is 0
        mask_img = Image.fromarray(mask)
        
        # Resize mask to match original image size
        if mask_img.size != img_size:
            mask_img = mask_img.resize(img_size, Image.Resampling.NEAREST)

        # Create output directories if needed
        os.makedirs(os.path.dirname(cloth_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)

        # save mask
        mask_img.save(mask_out_path)

        # save cloth image (RGB with transparent background)
        img_rgb = img.copy()
        img_rgb.putalpha(mask_img)
        img_rgb.save(cloth_out_path)

        print(f"✓ Cloth saved → {cloth_out_path}")
        print(f"✓ Mask saved → {mask_out_path}")


# Global extractor instance
_extractor = None


def extract(img_path, cloth_out_path, mask_out_path, device="cpu"):
    """Extract cloth from image using SegFormer"""
    global _extractor
    if _extractor is None:
        _extractor = SegformerClothExtractor(device=device)
    _extractor.extract(img_path, cloth_out_path, mask_out_path)

