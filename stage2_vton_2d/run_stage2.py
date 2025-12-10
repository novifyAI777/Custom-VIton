import os
import subprocess

def run_stage2(stage1_dir, out_dir):
    repo = "stage2_vton_2d/cp-vton-plus"
    dataset = "../cp_vton_pp/dataset"
    pairs = "test_pairs.txt"

    print("### Stage 2 — CP-VITON++ Inference")

    # 1️⃣ Prepare dataset
    from prepare_inputs import prepare
    prepare(stage1_dir, "stage2_vton_2d/cp_vton_pp/dataset")

    # 2️⃣ Run GMM (warping)
    print("### Running GMM (Geometric Matching Module)...")
    cmd_gmm = [
        "python", "test.py",
        "--dataroot", dataset,
        "--datamode", "test",
        "--data_list", pairs,
        "--stage", "GMM",
        "--result_dir", os.path.join(out_dir, "gmm"),
        "--checkpoint", "./checkpoints/GMM.pth"
    ]
    subprocess.run(cmd_gmm, cwd=repo, check=True)

    # 3️⃣ Run TOM (try-on)
    print("### Running TOM (Try-On Module)...")
    cmd_tom = [
        "python", "test.py",
        "--dataroot", dataset,
        "--datamode", "test",
        "--data_list", pairs,
        "--stage", "TOM",
        "--result_dir", out_dir,
        "--checkpoint", "./checkpoints/TOM.pth"
    ]
    subprocess.run(cmd_tom, cwd=repo, check=True)

    print("### CP-VTON++ COMPLETE ✔")
    

if __name__ == "__main__":
    import sys
    stage1 = sys.argv[1]
    out = sys.argv[2]
    run_stage2(stage1, out)
