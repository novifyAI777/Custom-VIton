# Stage 2: 2D Virtual Try-On (VTON)

Converts preprocessed person images and cloth into realistic 2D try-on results using VITON-HD.

## Overview

**Input:** Preprocessed images from Stage 1
- Person image (cleaned, segmented)
- Cloth image (cleaned, segmented)
- Keypoints and masks

**Output:** Try-on images in two formats
- `tryon/` - Simple try-on results
- `aligned/` - Aligned try-on results (used by Stage 3)

## Architecture

VITON-HD provides:
- High-definition (1024×768) try-on results
- Realistic clothing deformation
- Body segmentation awareness
- Fast inference (~1-2 seconds per image)

## Usage

### From Pipeline
```bash
python pipeline/pipeline.py <person_image> <cloth_image> [work_dir]
```

### Direct Execution
```bash
python stage2_vton_2d/run_stage2.py <stage1_output_dir> <output_dir>
```

## Input Format

Expects Stage 1 output directory with:
```
stage1_output/
├── image/                 # Cleaned person images
├── cloth/                 # Cleaned cloth images  
├── image-parse/           # Segmentation masks
├── cloth-mask/            # Cloth masks
├── openpose-json/         # Body keypoints
└── test_pairs.txt         # Image pair mappings
```

## Output Format

```
stage2_output/
├── tryon/                 # Simple try-on results
│   └── *.png             # Try-on images
└── aligned/              # Aligned try-on results
    └── *.png             # Aligned try-on images (→ Stage 3 input)
```

## Configuration

### Models & Checkpoints
- VITON-HD checkpoint: `checkpoints/viton_hd.pth` (auto-downloaded)
- OpenPose keypoints: Used from Stage 1 output

### Parameters (in run_stage2.py)
- `image_height`: 768
- `image_width`: 1024  
- `norm_type`: "instance"
- `batch_size`: 1

## Dependencies

```
torch>=1.13.0
torchvision
numpy
Pillow
opencv-python
tqdm
omegaconf
```

See `requirements.txt` for complete list.

## Performance

- **Speed:** ~1-2 seconds per image on GPU
- **Memory:** ~4GB GPU required
- **Quality:** High-definition (1024×768)
- **Failure Rate:** <5% on typical inputs

## Known Limitations

1. **Cloth Shape:** Works best with regular clothing (t-shirts, dresses)
   - Complex/loose clothing may deform incorrectly
   
2. **Body Position:** Requires frontal or near-frontal poses
   - Side/back poses produce lower quality results

3. **Occlusion:** Cannot handle severe occlusions
   - Arms covering upper body affects quality

4. **Texture Transfer:** Limited to cloth texture preservation
   - Logos/patterns may not align perfectly

## Troubleshooting

### Out of Memory
```python
# Reduce batch size in run_stage2.py
batch_size = 1  # Already minimum
```

### Poor Try-On Quality
1. Check Stage 1 preprocessing
2. Verify body keypoints are correct
3. Ensure cloth image is clean and well-segmented

### Missing Keypoints
- IDOL/OpenPose may fail on unusual poses
- Try different angle/position of person

## Stage 2 → Stage 3 Connection

The `aligned/` output from Stage 2 serves as input to Stage 3:
```
data/stage2_output/aligned/
    ↓ (read by Stage 3)
data/stage3_output/
    (3D models)
```

## References

- **VITON-HD**: [Paper](https://arxiv.org/abs/2104.10831)
- **Dataset**: DeepFashion, Dressed-In-Order

## Related Files

- `run_stage2.py` - Main execution script
- `VITON-HD/` - VITON-HD implementation
- `datasets.py` - Data loading utilities
- `networks.py` - Model architecture
