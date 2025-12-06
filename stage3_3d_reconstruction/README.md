# Stage 3: 3D Reconstruction

Converts 2D virtual try-on results into 3D mesh models using PIFuHD.

## Overview

Stage 3 takes the 2D try-on image from Stage 2 and reconstructs a full 3D mesh model of the person wearing the garment.

## Components

### 1. PIFuHD Reconstruction (`pifuhd/`)
- **Main Implementation**: `pifuhd_infer.py`
- **Method**: Pixel-aligned Implicit Function for High-Resolution 3D Human Digitization
- **Input**: 2D image from Stage 2
- **Output**: 3D mesh (.obj file)

### 2. Optional PHORHuM (`phorhum_optional/`)
- Alternative 3D reconstruction method
- Currently placeholder implementation

## Current Status

⚠️ **Note**: The current implementation is a **placeholder**. For full 3D reconstruction:

1. **Integrate PIFuHD**:
   ```bash
   git clone https://github.com/facebookresearch/pifuhd
   cd pifuhd
   # Follow their installation instructions
   ```

2. **Download Models**:
   - Download PIFuHD pretrained weights
   - Place in `checkpoints/` directory

3. **Update Implementation**:
   - Replace placeholder in `pifuhd_infer.py` with actual PIFuHD inference code

## Usage

### Command Line

```bash
python stage3_3d_reconstruction/run_stage3.py <input_image> [output_dir]
```

### Python API

```python
from stage3_3d_reconstruction.run_stage3 import run_stage3

input_image = "data/stage2_output/tryon_result.png"
output_dir = "data/stage3_output"

mesh_path = run_stage3(input_image, output_dir)
```

## Input Requirements

- **Image**: 2D try-on result from Stage 2
- **Format**: PNG or JPG
- **Recommended Size**: 512x512 or 1024x1024

## Output

- **Format**: Wavefront OBJ (.obj)
- **Location**: `data/stage3_output/model.obj`
- **Contents**: 
  - Vertex positions
  - Texture coordinates
  - Face definitions
  - Normal vectors

## Dependencies

```bash
pip install torch torchvision
pip install pillow numpy
```

For full PIFuHD:
```bash
# Additional dependencies from PIFuHD repository
pip install trimesh scikit-image opencv-python
```

## File Structure

```
stage3_3d_reconstruction/
├── run_stage3.py              # Main entry point
├── pifuhd/
│   ├── pifuhd_infer.py       # PIFuHD inference
│   └── checkpoints/
│       └── pifuhd.pt         # Model weights
└── phorhum_optional/
    └── phorhum_infer.py      # Alternative method
```

## Integration with Pipeline

Stage 3 fits into the complete pipeline:

```
Stage 1 → Stage 2 → Stage 3 → Stage 4
(Preprocess) → (2D Try-on) → (3D Reconstruction) → (Export)
```

## Future Improvements

- [ ] Integrate full PIFuHD implementation
- [ ] Add PHORHuM as alternative method
- [ ] Support batch processing
- [ ] Add texture mapping
- [ ] Implement mesh optimization
- [ ] Add normal map generation

## References

- **PIFuHD**: [Facebook Research PIFuHD](https://github.com/facebookresearch/pifuhd)
- **Paper**: "PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization"

## Troubleshooting

### Common Issues

1. **Missing PIFuHD implementation**
   - Solution: Follow integration instructions above

2. **CUDA out of memory**
   - Solution: Reduce image resolution or use CPU mode

3. **Invalid mesh output**
   - Check input image quality
   - Verify model weights are loaded correctly

## Testing

Run tests:
```bash
pytest tests/test_stage3.py -v
```

## License

Follows the same license as the main Custom-VIton project.
