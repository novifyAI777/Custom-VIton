import os
import sys
from pifuhd.pifuhd_infer import reconstruct

def run_stage3(input_image, keypoints_json, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print("### Stage 3 — 3D Reconstruction (PIFuHD)")
    result_obj = reconstruct(input_image, keypoints_json, out_dir)
    print("3D mesh generated →", result_obj)


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "data/stage1_output/person_clean.png"
    kp  = sys.argv[2] if len(sys.argv) > 2 else "data/stage1_output/pose_keypoints.json"
    out = sys.argv[3] if len(sys.argv) > 3 else "data/stage3_output"

    run_stage3(img, kp, out)
