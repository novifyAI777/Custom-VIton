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
    output_dir = os.path.dirname(output_path)

    cmd = [
        "python", os.path.join(schp_dir, "simple_extractor.py"),
        "--dataset", "lip",
        "--model-restore", checkpoint,
        "--input-dir", input_dir,
        "--output-dir", output_dir,
    ]

    print("Running SCHP human parsing...")
    subprocess.run(cmd, check=True)
    
    # Find and rename the temp parsing file to final output
    temp_parsing_files = glob.glob(os.path.join(output_dir, '_temp_parsing_*.png'))
    if temp_parsing_files:
        # Use the first (and should be only) temp parsing file
        temp_file = temp_parsing_files[0]
        shutil.move(temp_file, output_path)
        
        # Clean up any other temp files
        for temp_file in temp_parsing_files[1:]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # Clean up any npy files that were created
    temp_npy_files = glob.glob(os.path.join(output_dir, '_temp_parsing_*.npy'))
    for npy_file in temp_npy_files:
        if os.path.exists(npy_file):
            os.remove(npy_file)
    
    print(f"Parsing saved → {output_path}")
