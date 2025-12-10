"""
Stage 1 Runner (Final Version)
Handles:
- Background removal (rembg)
- Human parsing (SCHP)
- Pose estimation (YOLOv8 Pose)
- Cloth extraction (SegFormer-B3 Fashion or other model)
"""

import os
import sys
import subprocess
import torch
from image_cleaning.rembg_clean import clean_background
from human_parsing.infer_parsing import run as run_parse
from pose_estimation.infer_pose import run as run_pose
from cloth_extraction.segformer_cloth import extract


def setup_cuda_environment():
    """Setup CUDA environment variables if CUDA is available"""
    if torch.cuda.is_available():
        # Try to find CUDA installation
        cuda_home_candidates = [
            '/usr/local/cuda',
            '/opt/cuda',
            '/usr/local/cuda-12',
            '/usr/local/cuda-11'
        ]
        
        for cuda_path in cuda_home_candidates:
            if os.path.exists(cuda_path):
                os.environ['CUDA_HOME'] = cuda_path
                print(f"[CUDA] Set CUDA_HOME to: {cuda_path}")
                break
        
        if 'CUDA_HOME' not in os.environ:
            # Fallback: try to get from nvcc
            try:
                nvcc_path = subprocess.check_output(['which', 'nvcc'], text=True).strip()
                cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
                os.environ['CUDA_HOME'] = cuda_home
                print(f"[CUDA] Set CUDA_HOME from nvcc: {cuda_home}")
            except:
                print("[WARNING] CUDA is available but CUDA_HOME could not be determined")
    else:
        print("[WARNING] CUDA is not available")


def run_all(person, cloth, outdir):
    """Run all preprocessing steps"""
    print("=" * 30)
    print("STAGE 1 - PREPROCESSING")  
    print("=" * 30)
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    # Setup CUDA environment before running human parsing
    setup_cuda_environment()
    
    # Step 1: Clean person image with rembg
    print("\nStep 1: Image Cleaning (rembg)")
    person_clean = os.path.join(outdir, "person_clean.png")
    person_mask = os.path.join(outdir, "person_mask.png") 
    clean_background(person, person_clean, person_mask)
    print(f"[OK] {os.path.basename(person_clean)}, {os.path.basename(person_mask)} created")
    
    # Step 2: Human parsing  
    print("\nStep 2: Human Parsing (SCHP)")
    run_parse(person_clean, outdir)  # Pass outdir directly instead of parsing_out
    
    # Verify the parsing output exists
    parsing_file = os.path.join(outdir, "parsing.png")
    if os.path.exists(parsing_file):
        print(f"[OK] parsing.png created at {parsing_file}")
    else:
        print("[ERROR] parsing.png was not created")
            
    
    # Step 3: Generate openpose keypoints
    print("\nStep 3: OpenPose Keypoints")
    pose_out = os.path.join(outdir, "pose.json")
    run_pose(person_clean, pose_out)
    print(f"[OK] {os.path.basename(pose_out)} created")
    
    # Step 4: Cloth extraction
    print("\nStep 4: Cloth Extraction")
    cloth_clean_out = os.path.join(outdir, "cloth_clean.png")
    cloth_mask_out = os.path.join(outdir, "cloth_mask.png")
    extract(cloth, cloth_clean_out, cloth_mask_out)
    print(f"[OK] {os.path.basename(cloth_clean_out)}, {os.path.basename(cloth_mask_out)} created")
    
    print(f"\nAll preprocessing completed! Check {outdir}")


if __name__ == "__main__":
    person = sys.argv[1] if len(sys.argv) > 1 else "data/input/person.png"
    cloth = sys.argv[2] if len(sys.argv) > 2 else "data/input/dress.png"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "data/stage1_output"

    run_all(person, cloth, outdir)
