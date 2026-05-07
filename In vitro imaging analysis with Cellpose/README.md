# Cellpose Segmentation and Single-Cell Ratio Analysis

## Overview
This workflow performs Cellpose-based segmentation and single-cell fluorescence ratio analysis on pre-computed MIP TIFF images.

It extracts per-cell intensities from two channels and computes fluorescence ratios (e.g., `ch01 / ch00`) across experimental conditions.

---

## Features
- Cellpose segmentation on selected channel
- Mask and overlay image generation
- Single-cell intensity extraction
- Per-cell ratio calculation
- Filtering of low-quality measurements
- Export of CSV tables and plots

---

## Repository Structure

### `example_MIP_images/`
Input MIP images used for analysis.

- Contains paired channel images (e.g., `ch00`, `ch01`)
- Organized by condition, pH and field of view

### `example_masks_MIP_images/`
Example segmentation outputs.

- Cellpose-generated masks and overlay images
- Used for validation and reference

### `output_cells_analysis/`
Final analysis outputs.

- Per-cell ratio tables
- Filtered results and summary plots

### `Cellpose_segmentation&analysis.ipynb`
Main notebook for running segmentation and single-cell ratio analysis.

---

## Requirements
Install required packages:

```bash
pip install numpy pandas matplotlib tifffile scikit-image cellpose
```

Recommended:
- Python 3.9+
- Jupyter Notebook or JupyterLab
- GPU (optional, for faster segmentation)

---

## Input Data
- Pre-computed MIP TIFF images
- Two aligned channels per field: 

e.g.
  - `ch00_MIP.tif`
  - `ch01_MIP.tif`


Folder structure:

```text
├── condition1/
│   ├── v1/
│   │   ├──ph_n1
│   │   │   ├── ch00_MIP.tif
│   │   │   └── ch01_MIP.tif
│   │   ├──ph_n2
│   │       ├── ch00_MIP.tif
│   │       └── ch01_MIP.tif
│   ├── v2/
│   │   ├──ph_n1
│   │       ├── ch00_MIP.tif
│   │       └── ch01_MIP.tif
│   └── ...
└── condition2/
│   ├── v1/
│   │   ├──ph_n1
│   │   │   ├── ch00_MIP.tif
│   │   │   └── ch01_MIP.tif
│   │   ├──ph_n2
│   │       ├── ch00_MIP.tif
│   │       └── ch01_MIP.tif
│   ├── v2/
│   │   ├──ph_n1
│   │       ├── ch00_MIP.tif
│   │       └── ch01_MIP.tif
│   └── ...
```

Requirements:
- Only `condition1` and `condition2` are used
- Each field of view must contain both channels
- Images must already be MIPs
- Images must be aligned and same size

---

## Workflow

### 1. Load Images
Import MIP TIFF images for both channels.

### 2. Segmentation
- Run Cellpose on selected channel (default: `ch00`) using the Cellpose segmentation algorithm (Stringer et al., 2021 [1])
- Generate:
  - Cell masks (`.npy`)
  - Overlay images (`.png`)

### 3. Intensity Extraction
For each segmented cell:
- Compute mean intensity in:
  - `ch00`
  - `ch01`
- Store metadata:
  - condition
  - field of view
  - cell ID
  - channel intensities
  - computed ratio

### 4. Ratio Calculation
- Compute per-cell ratio:
  - `ch01 / ch00`
- Configurable numerator and denominator channels

### 5. Filtering
Exclude cells with:
- Low SBR denominator 
- Zero or negative denominator intensity
- Saturated signal in either channel
- Invalid ratio values

---

## Output
Saved in `output/`:

- `single_cell_ratios.csv`
- `single_cell_raw_before_filtering.csv`
- `filtering_summary.csv`
- `ratio_distributions.png`

Masks and overlays are saved in a mirrored folder structure:

```text
example_MIP_images/
├── condition1_mask/
│   ├── v1/
│   │   ├──ph_n1
│   │   │   ├── ch00_labels.npy
│   │   │   └── ch00_overlay.png
│   │   ├──ph_n2
│   │   │   ├── ch00_labels.npy
│   │   │   └── ch00_overlay.png
│   ├── v2/
│   └── ...
└── condition2_mask/
    ├── v1/
│   │   ├──ph_n1
│   │   │   ├── ch00_labels.npy
│   │   │   └── ch00_overlay.png
    ├── v2/
    └── ...
```

---

## Parameters to Adjust
When applying to new datasets:
- Input folder (`root_dir`)
- Mask output folder (`output_mask_dir`)
- Results output folder (`output_results_dir`)
- Conditions (`conditions`)
- Segmentation channel (`segment_channel`)
- Ratio numerator (`ratio_num`)
- Ratio denominator (`ratio_den`)
- GPU usage (`use_gpu`)
- Cell diameter (`diameter`)
- Filtering thresholds

Keep analysis settings consistent for cross-condition comparison.

---

## Notes
- Input images must already be maximum-intensity projectionsvisu
- No Z-stack processing is performed
- Folder structure must be consistent
- Each field of view must contain both channels
- Segmentation overlays must be visually checked before analysis
- Filtering must be perfomed carefully as it strongly affects final ratio results
- Keep plotting and filtering settings consistent across experiments

---

## Reference

1. Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. Nature Methods. 2021;18:100–106.