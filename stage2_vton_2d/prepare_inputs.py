"""
Auto-generated template file
Prepare inputs for VITON
"""

import os
from PIL import Image
import shutil

def prepare(person_clean, cloth_clean, parsing, pose, out_dir, size=(1024,1024)):
    os.makedirs(out_dir, exist_ok=True)
    p = Image.open(person_clean).convert('RGB').resize(size)
    c = Image.open(cloth_clean).convert('RGBA').resize(size)
    p.save(os.path.join(out_dir,'person_viton.png'))
    c.save(os.path.join(out_dir,'cloth_viton.png'))
    # copy parsing and pose as-is
    shutil.copy(parsing, os.path.join(out_dir,'parsing.png'))
    shutil.copy(pose, os.path.join(out_dir,'pose_keypoints.json'))
    print('Prepared inputs for VITON at', out_dir)
