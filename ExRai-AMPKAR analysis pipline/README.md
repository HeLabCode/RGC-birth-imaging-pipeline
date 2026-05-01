# Per-View Ratiometric Analysis Pipeline

## Overview
This pipeline performs cell segmentation, tracking, and ratiometric signal extraction from two-photon imaging datasets acquired with alternating excitation wavelengths (800 nm and 920 nm).

It is designed for experiments where the same fluorescent reporter is measured under two excitation conditions and compared at the single-cell level.

---

## Features
- Neural-network and manual-assisted cell segmentation
- Cell tracking across Z-stacks
- Inter-channel cell matching (800 nm vs 920 nm)
- Illumination quality filtering
- Local background subtraction
- Size-consistency filtering
- Final per-cell ratio computation
- Pseudo-colored visualization output

---

## Requirements

```bash
pip install numpy scipy scikit-image opencv-python tifffile matplotlib torch segmentation-models-pytorch ipywidgets trackpy
```

---

## Input Data

Each dataset ("view") must contain:
- Paired TIFF stacks:
  - `800_*.tif`
  - `920_*.tif`

Requirements:
- Same field of view
- Same Z-depth
- Preprocessed (renamed and paired)

---

## Workflow

### S0 – Visualization
- Load paired stacks
- Inspect alignment and image quality

### S1 – Segmentation
**S1.1 Neural Network**
- Automatic segmentation using pretrained model

**S1.2 Manual Refinement**
- Interactive region growing
- Merge/split cells
- Save final masks

---

### S2 – Z-Tracking
- Align slices using manual anchor points
- Track cells across Z-stack

---

### S3 – Channel Mapping
- Match cells between 800 nm and 920 nm
- Interactive validation
- Hungarian or nearest-neighbor matching

---

### S4 – Illumination Filtering
- Detect well-illuminated regions
- Remove unreliable cells

---

### S5 – Background Subtraction
- Compute local background per cell
- Generate corrected intensity values

---

### S6 – Size Filtering
- Remove mismatched cell pairs
- Based on mask size consistency

---

### S7 – Final Output
- Compute excitation ratio per cell
- Generate:
  - Results table
  - Pseudo-colored projection image

---

## Output
- Per-cell ratio table
- Visualization images
- Intermediate QC files
- Saved processing states

All outputs are stored in a `results/` directory within the dataset folder.

---

## Key Parameters
Adjust when applying to new data:
- Segmentation thresholds
- Tracking alignment points
- Matching distance thresholds
- Illumination cutoff levels
- Size ratio limits

---

## Notes
- Manual steps are required for segmentation, tracking, and mapping
- Quality control at each stage is critical
- Consistent parameter settings are required for cross-sample comparison
