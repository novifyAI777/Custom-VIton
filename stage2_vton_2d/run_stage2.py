import os
import subprocess
import shutil
import torch

def run_stage2(stage1_dir, out_dir):
    repo = "stage2_vton_2d/cp-vton-plus"
    dataset = "../cp_vton_pp/dataset"
    pairs = "test_pairs.txt"

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    print(f"### Stage 2 — CP-VITON++ Inference (GPU: {gpu_available})")

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
    env = os.environ.copy()
    if gpu_available:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd_gmm, cwd=repo, check=True, env=env)
    
    # Copy GMM outputs to dataset for TOM to use
    print("### Copying GMM outputs to dataset...")
    # The GMM output is saved relative to the cp-vton-plus directory
    # result_dir is 'data/stage2_output/gmm' (relative to cp-vton-plus)
    # So the full path from root is: stage2_vton_2d/cp-vton-plus/data/stage2_output/gmm/GMM/test/
    gmm_output_base = os.path.join(repo, "data", "stage2_output", "gmm", "GMM", "test")
    gmm_warp_cloth_dir = os.path.join(gmm_output_base, "warp-cloth")
    gmm_warp_mask_dir = os.path.join(gmm_output_base, "warp-mask")
    
    dataset_base = os.path.abspath("stage2_vton_2d/cp_vton_pp/dataset/test")
    dataset_warp_cloth_dir = os.path.join(dataset_base, "warp-cloth")
    dataset_warp_mask_dir = os.path.join(dataset_base, "warp-mask")
    
    os.makedirs(dataset_warp_cloth_dir, exist_ok=True)
    os.makedirs(dataset_warp_mask_dir, exist_ok=True)
    
    # Copy all warped cloth and mask files
    if os.path.exists(gmm_warp_cloth_dir):
        for file in os.listdir(gmm_warp_cloth_dir):
            src = os.path.join(gmm_warp_cloth_dir, file)
            dst = os.path.join(dataset_warp_cloth_dir, file)
            shutil.copy2(src, dst)
            print(f"  Copied: {src} -> {dst}")
    else:
        print(f"  Warning: GMM warp-cloth directory not found: {gmm_warp_cloth_dir}")
    
    if os.path.exists(gmm_warp_mask_dir):
        for file in os.listdir(gmm_warp_mask_dir):
            src = os.path.join(gmm_warp_mask_dir, file)
            dst = os.path.join(dataset_warp_mask_dir, file)
            shutil.copy2(src, dst)
            print(f"  Copied: {src} -> {dst}")
    else:
        print(f"  Warning: GMM warp-mask directory not found: {gmm_warp_mask_dir}")

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
    subprocess.run(cmd_tom, cwd=repo, check=True, env=env)

    # 4️⃣ Copy TOM outputs to final destination
    print("### Copying TOM outputs to final destination...")
    tom_output_dir = os.path.join(repo, "data", "stage2_output", "gmm", "test", "try-on")
    
    final_output_dir = os.path.abspath(out_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    
    if os.path.exists(tom_output_dir):
        for file in os.listdir(tom_output_dir):
            src = os.path.join(tom_output_dir, file)
            dst = os.path.join(final_output_dir, file)
            shutil.copy2(src, dst)
            print(f"  Copied TOM result: {src} -> {dst}")
    else:
        print(f"  Warning: TOM result directory not found: {tom_output_dir}")

    print("### CP-VTON++ COMPLETE ✔")
    

if __name__ == "__main__":
    import sys
    stage1 = sys.argv[1]
    out = sys.argv[2]
    run_stage2(stage1, out)
