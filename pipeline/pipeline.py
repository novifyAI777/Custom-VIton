"""
Auto-generated template file
Orchestrator pipeline
Place model weights in the models/ folder and implement loading as needed.
"""

import os
from stage1_preprocessing.run_stage1 import run_all as run_stage1
from stage2_vton_2d.run_stage2 import run_stage2
from stage3_3d_reconstruction.run_stage3 import run_stage3
from stage4_postprocess_export.run_stage4 import run_stage4

def run_pipeline(person_img, cloth_img, work_dir):
    print('Starting pipeline...')
    s1_out = os.path.join(work_dir, 'stage1_output')
    s2_out = os.path.join(work_dir, 'stage2_output')
    s3_out = os.path.join(work_dir, 'stage3_output')
    s4_out = os.path.join(work_dir, 'stage4_output')

    run_stage1(person_img, cloth_img, s1_out)
    run_stage2(s1_out, s2_out)
    run_stage3(s2_out, s3_out)
    run_stage4(s3_out, s4_out)
    print('Pipeline finished. Final outputs in', s4_out)

if __name__ == '__main__':
    import sys
    person = sys.argv[1] if len(sys.argv)>1 else '../data/input/person.jpg'
    dress = sys.argv[2] if len(sys.argv)>2 else '../data/input/dress.jpg'
    work = sys.argv[3] if len(sys.argv)>3 else '../data'
    run_pipeline(person, dress, work)
