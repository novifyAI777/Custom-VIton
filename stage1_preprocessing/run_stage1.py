"""
Stage 1 Runner (Final Version)
Handles:
- Background removal (rembg)
- Human parsing (SCHP)
- Pose estimation (YOLOv8 Pose)
- Cloth extraction (SegFormer-B3 Fashion or other model)
"""

import os
from image_cleaning.rembg_clean import clean_background
from human_parsing.infer_parsing import run as run_parse
from pose_estimation.infer_pose import run as run_pose
from cloth_extraction.segformer_cloth import extract


def run_all(input_person, input_cloth, out_dir):
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # define output file paths
    person_clean = os.path.join(out_dir, "person_clean.png")
    person_mask = os.path.join(out_dir, "person_mask.png")
    parsing_out = os.path.join(out_dir, "parsing.png")
    pose_out = os.path.join(out_dir, "pose_keypoints.json")
    cloth_clean_out = os.path.join(out_dir, "cloth_clean.png")
    cloth_mask_out = os.path.join(out_dir, "cloth_mask.png")

    print("\n==============================")
    print("STAGE 1 - PREPROCESSING")
    print("==============================\n")

    # Background removal (rembg)
    print("Step 1: Image Cleaning (rembg)")
    clean_background(input_person, person_clean, person_mask)
    print("[OK] person_clean.png, person_mask.png created\n")

    # Human parsing (SCHP)
    print("Step 2: Human Parsing (SCHP)")
    run_parse(person_clean, parsing_out)
    print("[OK] parsing.png created\n")

    # Pose extraction (YOLOv8-Pose)
    print("Step 3: Pose Estimation (YOLOv8-Pose)")
    run_pose(person_clean, pose_out)
    print("[OK] pose_keypoints.json created\n")

    # Cloth extraction (SegFormer)
    print("Step 4: Cloth Extraction")
    extract(input_cloth, cloth_clean_out, cloth_mask_out)
    print("[OK] cloth_clean.png, cloth_mask.png created\n")

    print("Stage 1 Completed Successfully!")
    print("Output Directory:", out_dir)


if __name__ == "__main__":
    import sys
    
    person = sys.argv[1] if len(sys.argv) > 1 else "data/input/person.png"
    cloth = sys.argv[2] if len(sys.argv) > 2 else "data/input/dress.png"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "data/stage1_output"

    run_all(person, cloth, outdir)
