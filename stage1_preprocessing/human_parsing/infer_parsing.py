"""
Wrapper for SCHP human parsing model
"""

import os
import subprocess
import shutil
import glob
from pathlib import Path

def run(input_image, output_path):
    schp_dir = os.path.join(os.path.dirname(__file__), "schp")
    checkpoint = os.path.join(schp_dir, "checkpoints", "latest.pth")
    
    input_dir = os.path.dirname(input_image)
    input_filename = os.path.basename(input_image)
    
    # If output_path is a directory, create the full path
    if os.path.isdir(output_path) or not output_path.endswith('.png'):
        output_dir = output_path
        final_output_path = os.path.join(output_dir, "parsing.png")
    else:
        output_dir = os.path.dirname(output_path)
        final_output_path = output_path

    cmd = [
        "python", os.path.join(schp_dir, "simple_extractor.py"),
        "--dataset", "lip",
        "--model-restore", checkpoint,
        "--input-dir", input_dir,
        "--output-dir", output_dir,
    ]

    print("Running SCHP human parsing...")
    print(f"[DEBUG] Input dir: {input_dir}")
    print(f"[DEBUG] Output dir: {output_dir}")
    print(f"[DEBUG] Input image: {input_image}")
    print(f"[DEBUG] Checkpoint: {checkpoint}")
    
    # List files in input directory
    if os.path.exists(input_dir):
        input_files = os.listdir(input_dir)
        print(f"[DEBUG] Files in input dir: {input_files}")
    else:
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] SCHP failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return
    
    # Find and rename the temp parsing file to final output
    temp_parsing_files = glob.glob(os.path.join(output_dir, '*_temp_parsing_*.png'))
    if not temp_parsing_files:
        # Try without leading underscore
        temp_parsing_files = glob.glob(os.path.join(output_dir, 'temp_parsing_*.png'))
    
    if temp_parsing_files:
        # Use the first (and should be only) temp parsing file
        temp_file = temp_parsing_files[0]
        shutil.move(temp_file, final_output_path)
        print(f"✓ Moved {temp_file} → {final_output_path}")
        
        # Clean up any other temp files
        for temp_file in temp_parsing_files[1:]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        print(f"[WARNING] No temp parsing files found in {output_dir}")
        # List all files in output directory for debugging
        all_files = os.listdir(output_dir)
        print(f"[DEBUG] Files in output directory: {all_files}")
    
    # Clean up any npy files that were created
    temp_npy_files = glob.glob(os.path.join(output_dir, '*temp_parsing_*.npy'))
    for npy_file in temp_npy_files:
        if os.path.exists(npy_file):
            os.remove(npy_file)
    
    if os.path.exists(final_output_path):
        print(f"✓ Parsing saved → {final_output_path}")
    else:
        print(f"[ERROR] Parsing file not created at {final_output_path}")

