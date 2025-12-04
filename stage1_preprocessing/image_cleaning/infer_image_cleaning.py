"""
Auto-generated template file
Stage1 image cleaning wrapper
"""

from PIL import Image
import os
from u2net.u2net_infer import U2NetPredictor
import numpy as np

def run(input_path, out_image_path, out_mask_path, model_path=None):
    img = Image.open(input_path).convert('RGB')
    pred = U2NetPredictor(model_path)
    mask = pred.predict(img)
    pred.save_mask(mask, out_mask_path)
    # apply mask to image and save
    im_arr = np.array(img).astype('uint8')
    mask_3 = (mask[...,None]*255).astype('uint8')
    out = (im_arr * (mask_3/255)).astype('uint8')
    Image.fromarray(out).save(out_image_path)
    print(f"Saved cleaned image to {out_image_path} and mask to {out_mask_path}")

if __name__ == '__main__':
    import sys
    inp = sys.argv[1] if len(sys.argv)>1 else '../../data/input/person.jpg'
    outi = sys.argv[2] if len(sys.argv)>2 else '../../data/stage1_output/person_clean.png'
    outm = sys.argv[3] if len(sys.argv)>3 else '../../data/stage1_output/person_mask.png'
    run(inp, outi, outm)
