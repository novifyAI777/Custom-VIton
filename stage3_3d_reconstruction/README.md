# Stage 3: 3D Reconstruction

Converts 2D try-on images into 3D models using either IDOL or TripoSR.

## Overview

**Input:** 2D try-on images from Stage 2
- Location: `data/stage2_output/aligned/`
- Format: PNG/JPG images (1024×768)

**Output:** 3D mesh models
- Location: `data/stage3_output/`
- Formats: OBJ, GLB (depending on backend)

## Backends

### 🟢 TripoSR (Recommended for CPU)
- **Status:** ✓ Fully working on CPU
- **Model:** Feedforward single-image 3D reconstruction
- **Speed:** 2-5 minutes per image (CPU)
- **Quality:** Good for general objects, limited for humans
- **Device:** CPU compatible
- **Output:** OBJ/GLB mesh with textures

**Pros:**
- Fast and lightweight
- Works on CPU
- Good generalization to various objects
- Automatic background removal

**Cons:**
- Lower quality for human figures
- Single-image limitation
- May produce incomplete details

### 🔴 IDOL (GPU-required)
- **Status:** GPU-only (CUDA required)
- **Model:** 3D Gaussian Splatting with body fitting
- **Speed:** ~10-30 seconds per image (GPU)
- **Quality:** Excellent for humans with clothing
- **Device:** NVIDIA GPU required
- **Output:** PLY file (3D Gaussian Splatting format)

**Pros:**
- Specialized for human reconstruction
- Captures body/clothing details
- Fast on GPU
- Better quality for fashion/humans

**Cons:**
- Requires CUDA GPU
- Cannot run on CPU
- Slower initial setup

## Configuration

### Switch Backend

Edit [pipeline.py](pipeline.py) line 53:

```python
# For TripoSR (CPU)
STAGE3_BACKEND = "triposr"

# For IDOL (GPU only)
STAGE3_BACKEND = "idol"
```

### Direct Execution

**TripoSR (batch process all aligned images):**
```bash
python stage3_3d_reconstruction/run_stage3.py --batch
```

**TripoSR (single image):**
```bash
python stage3_3d_reconstruction/run_stage3.py --input path/to/image.png
```

**IDOL (requires GPU):**
```bash
cd stage3_3d_reconstruction/IDOL
bash scripts/pip_install.sh
bash scripts/download_files.sh
python run_demo.py --input path/to/image.png
```

## Input/Output Format

### Input
```
data/stage2_output/aligned/
├── image_1.png
├── image_2.png
└── ...
```

### Output (TripoSR)
```
data/stage3_output/
├── image_1.obj
├── image_2.obj
└── ...
```

### Output (IDOL)
```
data/stage3_output/
├── image_1.ply
├── image_2.ply
└── ...
```

## Usage Guide

### Complete Pipeline (Auto-selects backend)
```bash
python pipeline/pipeline.py person.jpg cloth.jpg
```

### TripoSR Only (CPU-friendly)
```bash
python stage3_3d_reconstruction/run_stage3.py --batch --output data/stage3_output
```

### IDOL Only (GPU required)
```bash
# Setup (one time)
cd stage3_3d_reconstruction/IDOL
bash scripts/pip_install.sh
bash scripts/download_files.sh

# Run
python run_demo.py --input data/stage2_output/aligned/image.png \
                   --output data/stage3_output
```

## Dependencies

### TripoSR
```
torch
transformers>=4.30.0
trimesh
Pillow
numpy
rembg
einops
```

### IDOL (GPU-only)
```
torch>=2.0 (with CUDA)
pytorch3d
omegaconf
hydra-core
lpips
smplx
```

## Performance Comparison

| Feature | TripoSR | IDOL |
|---------|---------|------|
| **Device** | CPU ✓ | GPU only |
| **Speed (per image)** | 2-5 min | 10-30 sec |
| **Quality (humans)** | Fair | Excellent |
| **Quality (objects)** | Good | N/A |
| **Texture** | Auto-generated | From input |
| **Setup** | Simple | Complex |
| **Installation** | Easy | Medium |

## Quality Considerations

### TripoSR Quality
- **Best for:** Objects, simple clothing, general scenes
- **Fair for:** Human figures, complex clothing
- **Limitations:** 
  - Single-image inference
  - May miss fine details
  - Background artifacts possible

### IDOL Quality
- **Best for:** Human figures with clothing (fashion)
- **Excellent for:** Fitting try-on results to body
- **Requirements:** 
  - GPU (CUDA 11.8+)
  - SMPL-X body model
  - Sufficient memory (8GB+ recommended)

## Troubleshooting

### TripoSR Issues

**Out of memory:**
```python
# Reduce resolution in run_stage3.py
mc_resolution = 128  # Lower than 256
```

**Poor mesh quality:**
- Ensure Stage 2 output has good alignment
- Check if background removal worked
- Try different foreground_ratio (0.75-0.95)

### IDOL Issues

**CUDA not available:**
```
❌ IDOL requires GPU with CUDA support
✓ Solution: Use TripoSR instead
```

**pytorch3d import error:**
```
Missing pytorch3d (GPU library only)
✓ Solution: Use TripoSR instead
```

**Checkpoint not found:**
```bash
cd stage3_3d_reconstruction/IDOL
bash scripts/download_files.sh
```

## CPU vs GPU Decision

**Use TripoSR if:**
- ✓ You have no GPU
- ✓ You want quick results (2-5 min per image)
- ✓ You're processing objects or simple scenes
- ✓ You want simple setup

**Use IDOL if:**
- ✓ You have NVIDIA GPU with CUDA
- ✓ You want high-quality human reconstruction
- ✓ You're doing fashion/try-on focused work
- ✓ Speed is critical (10-30 sec per image)

**Recommended:** TripoSR for most users (works on CPU)

## Output Visualization

### View Generated Models

```bash
# Gradio viewer (if available)
python stage3_3d_reconstruction/view_stage3_models.py
```

Or use external viewers:
- **Blender:** Import OBJ/GLB
- **Three.js:** Display in web
- **Meshlab:** Analyze/optimize mesh

## Model References

### TripoSR
- Paper: [TripoSR: Fast 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/2403.18181)
- Model: Stability AI
- Code: [github.com/VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)

### IDOL
- Paper: [IDOL: Implicit Dual-Occupancy Levels for High-Fidelity 3D Gaussian Splatting](https://arxiv.org/abs/2402.09478)
- Model: VAST AI Research
- Code: [github.com/VAST-AI-Research/IDOL](https://github.com/VAST-AI-Research/IDOL)

## Next Step: Stage 4

3D models from Stage 3 feed into Stage 4 (postprocessing & export):
```
data/stage3_output/
    ↓ (OBJ/PLY mesh)
data/stage4_output/
    (optimized GLB exports)
```

## Related Files

- `run_stage3.py` - Main execution script (both backends)
- `TripoSR/` - TripoSR implementation (clone from GitHub)
- `IDOL/` - IDOL implementation (clone from GitHub)
- `pipeline/pipeline.py` - Pipeline orchestrator (backend selector at line 53)
