"""
Auto-generated template file
Run stage2
"""

import os
from viton_hd.viton_infer import VITONHD
from prepare_inputs import prepare

def run_stage2(stage1_dir, out_dir, checkpoints=None):
    prepare(os.path.join(stage1_dir,'person_clean.png'),
            os.path.join(stage1_dir,'cloth_clean.png'),
            os.path.join(stage1_dir,'parsing.png'),
            os.path.join(stage1_dir,'pose_keypoints.json'),
            out_dir)
    model = VITONHD(checkpoints)
    model.infer(os.path.join(out_dir,'person_viton.png'),
                os.path.join(out_dir,'cloth_viton.png'),
                os.path.join(out_dir,'parsing.png'),
                os.path.join(out_dir,'pose_keypoints.json'),
                os.path.join(out_dir,'tryon_result.png'))

if __name__ == '__main__':
    import sys
    s1 = sys.argv[1] if len(sys.argv)>1 else '../data/stage1_output'
    out = sys.argv[2] if len(sys.argv)>2 else '../data/stage2_output'
    run_stage2(s1, out)
