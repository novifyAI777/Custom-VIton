"""
Stage 2: Virtual Try-On with VITON-HD
"""

import os
import sys
import subprocess

def run_stage2(stage1_dir, out_dir, checkpoints=None):
    print("\n### Stage 2 — Virtual Try-On (VITON-HD)")
    
    # Verify Stage 1 outputs exist
    required_files = ['person_clean.png', 'cloth_clean.png', 'cloth_mask.png', 
                      'parsing.png', 'pose_keypoints.json']
    for fname in required_files:
        fpath = os.path.join(stage1_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Step 1: Setup VITON-HD dataset structure
    print("\nStep 1: Preparing VITON-HD dataset...")
    from preprocess_viton import setup_viton_dataset
    
    dataset_dir = os.path.join('stage2_vton_2d', 'viton_hd_repo', 'datasets')
    setup_viton_dataset(stage1_dir, dataset_dir, img_size=(768, 1024))
    
    # Step 2: Run VITON-HD inference using the original test.py
    print("\nStep 2: Running VITON-HD inference...")
    
    if checkpoints is None:
        checkpoints = os.path.join('stage2_vton_2d', 'viton_hd_repo', 'checkpoints')
    
    # Change to viton_hd_repo directory
    viton_repo = os.path.join('stage2_vton_2d', 'viton_hd_repo')
    
    # Run the CPU-compatible VITON-HD test script
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    cmd = [
        'python', 'test_cpu.py',
        '--name', 'viton_output',
        '--dataset_dir', './datasets/',
        '--dataset_list', 'test_pairs.txt',
        '--checkpoint_dir', './checkpoints/',
        '--save_dir', os.path.abspath(out_dir),
        '--batch_size', '1',
        '--load_height', '1024',
        '--load_width', '768',
        '--device', device
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=viton_repo, check=True)
    
    # Find output file
    output_path = os.path.join(out_dir, 'viton_output', 'person.png')
    
    if os.path.exists(output_path):
        print(f"\n✔ Stage 2 Complete → {output_path}\n")
        return output_path
    else:
        print(f"\n⚠️  Output not found at expected location: {output_path}")
        print(f"   Check {out_dir} for results")
        return None

if __name__ == '__main__':
    s1 = sys.argv[1] if len(sys.argv) > 1 else 'data/stage1_output'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/stage2_output'
    run_stage2(s1, out)
