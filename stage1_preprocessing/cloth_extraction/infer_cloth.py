"""
Auto-generated template file
Cloth extraction wrapper
"""

from PIL import Image
from u2net_cloth.u2net_cloth_infer import U2NetCloth
import numpy as np

def run(cloth_path, out_image, out_mask, model_path=None):
    img = Image.open(cloth_path).convert('RGB')
    seg = U2NetCloth(model_path)
    mask = seg.segment(img)
    seg.save(mask, out_mask)
    arr = np.array(img).astype('uint8')
    mask3 = (mask[...,None]*255).astype('uint8')
    out = (arr * (mask3/255)).astype('uint8')
    Image.fromarray(out).save(out_image)
    print(f"Saved cloth image {out_image} and mask {out_mask}")

if __name__ == '__main__':
    import sys
    inp = sys.argv[1] if len(sys.argv)>1 else '../../data/input/dress.jpg'
    outi = sys.argv[2] if len(sys.argv)>2 else '../../data/stage1_output/cloth_clean.png'
    outm = sys.argv[3] if len(sys.argv)>3 else '../../data/stage1_output/cloth_mask.png'
    run(inp, outi, outm)
