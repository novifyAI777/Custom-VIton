"""
Stage 4: Post-processing and Export
Cleans up 3D mesh, converts formats, and prepares for visualization

Usage:
    python run_stage4.py <input_mesh> <output_dir>
    
Example:
    python run_stage4.py data/stage3_output/model.obj data/stage4_output
"""

import os
import sys


def run_stage4(input_mesh, out_dir, convert_to_glb=True, cleanup=True):
    """
    Run Stage 4: Post-processing and Export
    
    Args:
        input_mesh: Path to input 3D mesh (.obj)
        out_dir: Output directory
        convert_to_glb: Whether to convert OBJ to GLB format
        cleanup: Whether to clean up the mesh
        
    Returns:
        Dict with paths to output files
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Check input exists
    if not os.path.exists(input_mesh):
        raise FileNotFoundError(f"Input mesh not found: {input_mesh}")
    
    print(f"\n{'='*50}")
    print("STAGE 4 - POST-PROCESSING & EXPORT")
    print(f"{'='*50}")
    print(f"Input: {input_mesh}")
    print(f"Output Directory: {out_dir}")
    print()
    
    results = {}
    
    # Step 1: Mesh cleanup (optional)
    cleaned_mesh = input_mesh
    if cleanup:
        try:
            from mesh_cleanup.trimesh_clean import clean_mesh
            cleaned_mesh = os.path.join(out_dir, "model_cleaned.obj")
            print("Step 1: Cleaning mesh...")
            clean_mesh(input_mesh, cleaned_mesh)
            results['cleaned_obj'] = cleaned_mesh
            print(f"✓ Cleaned mesh → {cleaned_mesh}")
        except ImportError:
            print("⚠ Mesh cleanup skipped (trimesh not installed)")
            cleaned_mesh = input_mesh
    
    # Step 2: Convert to GLB format (optional)
    if convert_to_glb:
        try:
            from convert_obj_to_glb.obj2glb import convert_obj_to_glb
            glb_output = os.path.join(out_dir, "model.glb")
            print("\nStep 2: Converting to GLB format...")
            convert_obj_to_glb(cleaned_mesh, glb_output)
            results['glb'] = glb_output
            print(f"✓ GLB file → {glb_output}")
        except ImportError:
            print("⚠ GLB conversion skipped (required libraries not installed)")
    
    # Copy final OBJ to output
    if cleaned_mesh != input_mesh:
        results['final_obj'] = cleaned_mesh
    else:
        import shutil
        final_obj = os.path.join(out_dir, "model_final.obj")
        shutil.copy(input_mesh, final_obj)
        results['final_obj'] = final_obj
    
    print(f"\n{'='*50}")
    print("Stage 4 Complete!")
    print(f"Output files:")
    for key, path in results.items():
        print(f"  {key}: {path}")
    print(f"{'='*50}\n")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_stage4.py <input_mesh> [output_dir]")
        print("\nExample:")
        print("  python run_stage4.py data/stage3_output/model.obj data/stage4_output")
        sys.exit(1)
    
    input_mesh = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/stage4_output"
    
    run_stage4(input_mesh, output_dir)
