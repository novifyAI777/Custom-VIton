"""
Auto-generated template file
Run stage1 - image cleaning, parsing, pose, cloth extraction
"""

import os
from image_cleaning.infer_image_cleaning import run as run_clean
from human_parsing.infer_parsing import run as run_parse
from pose_estimation.infer_pose import run as run_pose
from cloth_extraction.infer_cloth import run as run_cloth

def run_all(input_person, input_cloth, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    person_clean = os.path.join(out_dir, 'person_clean.png')
    person_mask = os.path.join(out_dir, 'person_mask.png')
    parsing = os.path.join(out_dir, 'parsing.png')
    pose = os.path.join(out_dir, 'pose_keypoints.json')
    cloth_img = os.path.join(out_dir, 'cloth_clean.png')
    cloth_mask = os.path.join(out_dir, 'cloth_mask.png')

    print('Running image cleaning...')
    run_clean(input_person, person_clean, person_mask)
    print('Running human parsing...')
    run_parse(person_clean, parsing)
    print('Running pose estimation...')
    run_pose(person_clean, pose)
    print('Running cloth extraction...')
    run_cloth(input_cloth, cloth_img, cloth_mask)
    print('Stage1 complete. Outputs in', out_dir)

if __name__ == '__main__':
    import sys
    inp_person = sys.argv[1] if len(sys.argv)>1 else '../data/input/person.jpg'
    inp_cloth = sys.argv[2] if len(sys.argv)>2 else '../data/input/dress.jpg'
    out = sys.argv[3] if len(sys.argv)>3 else '../data/stage1_output'
    run_all(inp_person, inp_cloth, out)
