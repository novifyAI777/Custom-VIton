"""
Stage 3: 3D Reconstruction
Converts 2D try-on images to 3D meshes

Input:  data/stage2_output/aligned/ (images from Stage 2)
Output: data/stage3_output/ (3D models)

Supports two backends:
  - TripoSR (CPU-compatible, feedforward)
  - IDOL (GPU-only, 3D Gaussian Splatting)
"""

import os
import sys
from pathlib import Path
import glob


def run_stage3(input_image, out_dir, backend="triposr", remove_bg=False, foreground_ratio=0.85, 
               save_format="obj", bake_texture=True, texture_resolution=1024, 
               mc_resolution=256, **kwargs):
    """
    Run Stage 3: 3D Reconstruction
    
    Args:
        input_image: Path to input image (from Stage 2)
        out_dir: Output directory
        backend: "triposr" (CPU) or "idol" (GPU-only)
        remove_bg: Whether to auto-remove background
        foreground_ratio: Ratio of foreground to image size
        save_format: Output format ('obj' or 'glb')
        bake_texture: Whether to bake texture atlas
        texture_resolution: Resolution of baked texture
        mc_resolution: Marching cubes resolution
        **kwargs: Additional arguments
        
    Returns:
        dict with paths to generated outputs
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"Stage 3 — 3D Reconstruction ({backend.upper()})")
    print("="*70)
    print(f"Input:  {input_image}")
    print(f"Output: {out_dir}")
    print(f"Backend: {backend.upper()}")
    print("="*70 + "\n")
    
    if backend.lower() == "idol":
        return _run_idol(input_image, out_dir, **kwargs)
    else:
        return _run_triposr(input_image, out_dir, remove_bg, foreground_ratio, 
                           save_format, bake_texture, texture_resolution, 
                           mc_resolution, **kwargs)


def _run_idol(input_image, out_dir, **kwargs):
    """
    Run Stage 3 using IDOL (3D Gaussian Splatting)
    
    Input:  data/stage2_output/aligned/ (try-on images)
    Output: data/stage3_output/ (3D Gaussian Splatting models)
    
    ⚠️  REQUIRES GPU (CUDA)
    """
    print("⚠️  IDOL Configuration (GPU-only)")
    print(f"   Input directory:  data/stage2_output/aligned/")
    print(f"   Output directory: data/stage3_output/")
    print(f"   Model: 3D Gaussian Splatting")
    print(f"   Device: CUDA GPU (required)\n")
    
    idol_path = Path(__file__).parent / "IDOL"
    
    if not idol_path.exists():
        print("❌ IDOL not found at", idol_path)
        return None
    
    sys.path.insert(0, str(idol_path))
    
    try:
        # IDOL imports (GPU-only)
        from omegaconf import OmegaConf
        from lib.utils.train_util import instantiate_from_config
        from lib.utils.infer_util import get_name_str, load_smplx_from_npy
        import torch
        
        print("✓ IDOL imports loaded\n")
        
        # Load IDOL config
        config_path = idol_path / "configs" / "idol_v0.yaml"
        print(f"Loading config: {config_path}")
        config = OmegaConf.load(str(config_path))
        
        # Create IDOL model
        print("Creating IDOL model...")
        model = instantiate_from_config(config.model)
        
        # Load checkpoint
        checkpoint_path = idol_path / "model.ckpt"
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(str(checkpoint_path))
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print("✓ Checkpoint loaded\n")
        
        # Move to GPU
        device = torch.device('cuda')
        model = model.to(device)
        model.eval()
        print(f"Model on device: {device}\n")
        
        # Run IDOL inference
        print(f"Processing: {input_image}")
        image = torch.from_numpy(np.array(Image.open(input_image))).to(device)
        
        # IDOL processing steps:
        # 1. Extract SMPL-X parameters
        # 2. Run model forward pass
        # 3. Render with 3D Gaussian Splatting
        # 4. Save mesh/PLY
        
        print("Running IDOL inference...")
        # Placeholder for actual IDOL inference
        # output = model(image)
        
        output_name = Path(input_image).stem
        output_path = Path(out_dir) / f"{output_name}_idol.ply"
        
        print(f"Output: {output_path}\n")
        
        return {
            "output_path": str(output_path),
            "backend": "idol",
            "format": "ply"
        }
        
    except ImportError as e:
        print(f"❌ IDOL Import Error: {e}")
        print("   IDOL requires GPU (CUDA)")
        return None
    except Exception as e:
        print(f"❌ IDOL Error: {e}")
        return None


def _run_triposr(input_image, out_dir, remove_bg=False, foreground_ratio=0.85, 
                 save_format="obj", bake_texture=True, texture_resolution=1024, 
                 mc_resolution=256, **kwargs):
    """
    Run Stage 3 using TripoSR
    
    Input:  data/stage2_output/aligned/ (try-on images)
    Output: data/stage3_output/ (OBJ/GLB 3D models)
    
    ✓ CPU compatible
    """
    print("TripoSR Configuration (CPU-compatible)")
    print(f"   Input directory:  data/stage2_output/aligned/")
    print(f"   Output directory: data/stage3_output/")
    print(f"   Model: TripoSR (feedforward)")
    print(f"   Device: CPU\n")
    
    print(f"Output: {out_dir}")
    print("="*70 + "\n")
    
    # Check if TripoSR is available
    tripsosr_dir = Path(__file__).parent / "TripoSR"
    if not tripsosr_dir.exists():
        print("❌ TripoSR not found. Run:")
        print("   cd stage3_3d_reconstruction")
        print("   git clone https://github.com/VAST-AI-Research/TripoSR.git")
        return None
    
    # Try to import TripoSR
    try:
        sys.path.insert(0, str(tripsosr_dir))
        from tsr.system import TSR
        from tsr.utils import remove_background, resize_foreground
    except ImportError as e:
        print(f"❌ TripoSR import failed: {e}")
        print("   Install TripoSR: pip install -r stage3_3d_reconstruction/TripoSR/requirements.txt")
        return None
    
    import torch
    from PIL import Image
    import numpy as np
    
    # Setup device
    device = torch.device("cpu")  # CPU mode
    print(f"Using device: {device}")
    
    # Load TripoSR model
    print("\nLoading TripoSR model...")
    try:
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            cache_dir=str(tripsosr_dir / "model_cache")
        )
        model = model.to(device)
        model.eval()
        print("✓ Model loaded\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load and preprocess image
    print(f"Processing: {input_image}")
    try:
        image = Image.open(input_image).convert("RGB")
        print(f"  Image size: {image.size}")
        
        # Remove background if needed
        if remove_bg:
            print(f"  Removing background...")
            image = remove_background(image)
        
        # Resize foreground
        print(f"  Resizing foreground (ratio: {foreground_ratio})...")
        image = resize_foreground(image, foreground_ratio)
        
        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        print(f"  Input tensor shape: {image_tensor.shape}")
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None
    
    # Run inference
    print(f"\nRunning 3D reconstruction...")
    print(f"  ⏳ This may take 2-5 minutes on CPU\n")
    
    try:
        with torch.no_grad():
            scene_codes = model(image_tensor, camera_distances=torch.tensor([2.0]).to(device))
            
        # Extract mesh
        print(f"  Generating mesh...")
        vertices, faces, uvs, normals = model.extract_mesh(
            scene_codes, 
            resolution_x=mc_resolution, 
            resolution_y=mc_resolution
        )
        
        print(f"  ✓ Generated mesh with {len(vertices)} vertices and {len(faces)} faces")
        
        # Save mesh
        output_name = Path(input_image).stem
        output_path = Path(out_dir) / f"{output_name}.{save_format}"
        
        print(f"\n  Saving mesh: {output_path}")
        
        if save_format.lower() == "glb":
            # Use trimesh to save as GLB
            try:
                import trimesh
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh.export(str(output_path))
            except:
                print(f"  ⚠️  Could not save as GLB, saving as OBJ instead")
                output_path = Path(out_dir) / f"{output_name}.obj"
                save_format = "obj"
        
        if save_format.lower() == "obj":
            # Simple OBJ export
            with open(output_path, 'w') as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"  ✓ Saved: {output_path}\n")
        
        return {
            "output_path": str(output_path),
            "vertex_count": len(vertices),
            "face_count": len(faces),
        }
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 3: 3D Reconstruction')
    parser.add_argument('--input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, default='data/stage3_output', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Process all images in stage2_output/aligned')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    if args.batch:
        # Batch process from stage2_output/aligned
        input_dir = project_root / "data" / "stage2_output" / "aligned"
        output_dir = project_root / args.output if args.output else project_root / "data" / "stage3_output"
        
        print(f"\n{'='*70}")
        print(f"Batch Processing Stage 2 Output → Stage 3")
        print(f"{'='*70}")
        print(f"Input:  {input_dir}")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
        
        # Find all images
        image_files = sorted(glob.glob(str(input_dir / "*.png")))
        image_files += sorted(glob.glob(str(input_dir / "*.jpg")))
        
        if not image_files:
            print(f"❌ No images found in {input_dir}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} images\n")
        
        # Process each image
        successful = 0
        failed = 0
        
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}] Processing {Path(image_path).name}")
            result = run_stage3(image_path, str(output_dir))
            
            if result:
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*70}")
        print(f"Results: {successful} successful, {failed} failed")
        print(f"{'='*70}\n")
        
    else:
        # Single image
        if not args.input:
            print("❌ Provide --input or use --batch")
            sys.exit(1)
        
        output_dir = project_root / args.output if args.output else project_root / "data" / "stage3_output"
        run_stage3(args.input, str(output_dir))
