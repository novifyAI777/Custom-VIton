"""
Lightweight orchestrator for all stages.
This file adjusts sys.path so stage modules import without manual PYTHONPATH.
"""

import os
import sys
import glob

# Add stage roots (and a few inner tool dirs) so intra-stage imports resolve when invoked from repo root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
STAGE_DIRS = [
    os.path.join(PROJECT_ROOT, 'stage1_preprocessing'),
    os.path.join(PROJECT_ROOT, 'stage2_vton_2d'),
    os.path.join(PROJECT_ROOT, 'stage3_3d_reconstruction'),
    os.path.join(PROJECT_ROOT, 'stage4_postprocess_export'),
    os.path.join(PROJECT_ROOT, 'stage5_visualization'),
]

# Known nested modules that expect their own root on sys.path (SCHP, TripoSR, etc.).
EXTRA_DIRS = [
    os.path.join(PROJECT_ROOT, 'stage1_preprocessing', 'human_parsing', 'schp'),
    os.path.join(PROJECT_ROOT, 'stage1_preprocessing', 'cloth_extraction'),
    os.path.join(PROJECT_ROOT, 'stage1_preprocessing', 'image_cleaning'),
    os.path.join(PROJECT_ROOT, 'stage3_3d_reconstruction', 'TripoSR'),
]

for path in [PROJECT_ROOT] + STAGE_DIRS + EXTRA_DIRS:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from stage1_preprocessing.run_stage1 import run_all as run_stage1
from stage2_vton_2d.run_stage2 import run_stage2
from stage3_3d_reconstruction.run_stage3 import run_stage3
from stage4_postprocess_export.run_stage4 import run_stage4
from stage5_visualization.run_stage5 import create_viewer as run_stage5

def run_pipeline(person_img, cloth_img, work_dir):
    work_dir = os.path.abspath(work_dir)
    print('Starting pipeline...')
    s1_out = os.path.join(work_dir, 'stage1_output')
    s2_out = os.path.join(work_dir, 'stage2_output')
    s3_out = os.path.join(work_dir, 'stage3_output')
    s4_out = os.path.join(work_dir, 'stage4_output')
    s5_out = os.path.join(work_dir, 'stage5_visualization')

    run_stage1(person_img, cloth_img, s1_out)
    run_stage2(s1_out, s2_out)

    # Stage 3: Process all aligned images from stage2_output/aligned
    # Input:  data/stage2_output/aligned/ (try-on images)
    # Output: data/stage3_output/ (3D models)
    # Backend: idol (GPU-only) or triposr (CPU)
    aligned_dir = os.path.join(s2_out, 'aligned')
    aligned_images = sorted(glob.glob(os.path.join(aligned_dir, '*.png')))
    
    STAGE3_BACKEND = "idol"  # Change to "triposr" for CPU
    
    if not aligned_images:
        print(f"⚠️  No aligned images found in {aligned_dir}")
        print(f"    Skipping Stage 3 (3D reconstruction)")
        mesh_obj = None
    else:
        print(f"\n{'='*70}")
        print(f"Stage 3: Processing {len(aligned_images)} images from Stage 2")
        print(f"Backend: {STAGE3_BACKEND.upper()}")
        print(f"{'='*70}")
        print(f"Input:  {aligned_dir}")
        print(f"Output: {s3_out}")
        print(f"{'='*70}\n")
        
        mesh_obj = run_stage3(aligned_images[0], s3_out, backend=STAGE3_BACKEND)
    
    if mesh_obj is None:
        print("\n⚠️  Stage 3 skipped or failed")
        print("    Using Stage 2 output (aligned images) as final result")
        stage4_results = {'final_obj': aligned_images[0]}
    else:
        stage4_results = run_stage4(mesh_obj, s4_out)

    # Prefer GLB for visualization; fall back to final OBJ if GLB missing
    glb_path = stage4_results.get('glb') or stage4_results.get('final_obj')
    if not glb_path or not os.path.exists(glb_path):
        print(f"⚠️  Stage 4 did not produce a GLB/OBJ, using Stage 2 output")
        glb_path = aligned_images[0] if aligned_images else None
    
    if glb_path:
        viewer_html = run_stage5(glb_path, s5_out)
        print('Pipeline finished. Final outputs in', s4_out)
        print('Stage 5 viewer at', viewer_html)
    else:
        print("❌ Pipeline failed: No output to visualize")

if __name__ == '__main__':
    import sys
    person = sys.argv[1] if len(sys.argv)>1 else '../data/input/person.jpg'
    dress = sys.argv[2] if len(sys.argv)>2 else '../data/input/dress.jpg'
    work = sys.argv[3] if len(sys.argv)>3 else '../data'
    run_pipeline(person, dress, work)
