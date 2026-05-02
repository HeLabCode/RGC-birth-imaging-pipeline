# Cellpose Segmentation and Single-Cell Ratio Analysis (MIP-based workflow)

Jupyter workflow for segmentation and fluorescence ratio analysis on **pre-computed MIP images** using Cellpose.

This version works directly on folders containing already-generated MIP TIFF images organized by **condition only**.

---

## What this workflow does

1. Uses existing MIP images from `ch00` and `ch01`
2. Runs Cellpose segmentation on a selected channel
3. Saves masks and overlay images in a mirrored folder structure
4. Extracts single-cell fluorescence intensities
5. Computes per-cell fluorescence ratios, for example `ch01 / ch00`
6. Applies filtering to remove invalid or low-quality measurements
7. Exports CSV tables and final plots into an `output` folder

---

## Repository contents

```text
.
├── Cellpose_condition1_condition2_ratio_analysis_v2.ipynb   # Main analysis notebook
├── example_MIP_images/                                      # Input MIP images
├── example_masks_MIP_images/                                # Cellpose mask outputs
├── output/                                                  # CSV results and plots
└── README.md                                                # Project documentation
```

---

## Requirements

Install the required Python packages with:

```bash
pip install numpy pandas matplotlib seaborn tifffile scikit-image scipy opencv-python cellpose
```

Recommended environment:

- Python 3.9+
- Jupyter Notebook or JupyterLab
- GPU support optional, but recommended for faster Cellpose segmentation

---

## Input data format

The notebook expects **already-MIPed TIFF images**.

The input folder must be organized like this:

```text
example_MIP_images/
├── condition1/
│   ├── v1/
│   │   ├── ch00_MIP.tif
│   │   └── ch01_MIP.tif
│   ├── v2/
│   │   ├── ch00_MIP.tif
│   │   └── ch01_MIP.tif
│   └── ...
└── condition2/
    ├── v1/
    │   ├── ch00_MIP.tif
    │   └── ch01_MIP.tif
    ├── v2/
    │   ├── ch00_MIP.tif
    │   └── ch01_MIP.tif
    └── ...
```

Required rules:

- Only `condition1` and `condition2` are used.
- Each field of view must be inside a folder such as `v1`, `v2`, `v3`, etc.
- Each field of view must contain:
  - `ch00_MIP.tif`
  - `ch01_MIP.tif`
- Images must already be maximum-intensity projections.
- No pH folders are used.
- No Z-stack processing is performed.

---

## Output folder structure

### Cellpose masks and overlays

Cellpose outputs are saved in:

```text
example_masks_MIP_images/
├── condition1_mask/
│   ├── v1/
│   │   ├── ch00_labels.npy
│   │   └── ch00_overlay.png
│   ├── v2/
│   └── ...
└── condition2_mask/
    ├── v1/
    │   ├── ch00_labels.npy
    │   └── ch00_overlay.png
    ├── v2/
    └── ...
```

The mask folder mirrors the input folder structure.

---

### Final values and plots

All quantification tables and plots are saved in:

```text
output/
├── single_cell_ratios.csv
├── single_cell_raw_before_filtering.csv
├── filtering_summary.csv
└── ratio_distributions.png
```

---

## Main output files

| File | Description |
|---|---|
| `single_cell_ratios.csv` | Filtered single-cell ratio results |
| `single_cell_raw_before_filtering.csv` | Raw per-cell intensity values before filtering |
| `filtering_summary.csv` | Summary of retained and removed cells per image |
| `ratio_distributions.png` | Final ratio distribution plot comparing conditions |

---

## Analysis details

### 1. Cellpose segmentation

The notebook runs Cellpose on the selected segmentation channel.

Default:

```python
SEGMENT_CHANNEL = "ch00"
```

For each image, the notebook saves:

- Cell label mask: `ch00_labels.npy`
- Segmentation overlay: `ch00_overlay.png`

The overlay images should be checked manually before interpreting the ratio results.

---

### 2. Single-cell intensity extraction

For each segmented cell, the notebook extracts mean fluorescence intensity from:

- `ch00_MIP.tif`
- `ch01_MIP.tif`

Each cell receives metadata for:

- condition
- field of view
- cell label
- channel intensities
- computed ratio

---

### 3. Ratio calculation

Default ratio:

```text
ratio = ch01_mean / ch00_mean
```

Default settings:

```python
RATIO_NUM = "ch01"
RATIO_DEN = "ch00"
```

Change these variables in the notebook if the numerator and denominator channels need to be swapped.

---

### 4. Filtering

Cells are excluded if:

- denominator intensity is too low
- denominator intensity is zero or negative
- either channel is saturated
- the computed ratio is invalid

Filtered and unfiltered data are both saved.

---

## Configuration

Main variables to adjust inside the notebook:

```python
ROOT_DIR = "example_MIP_images"
OUTPUT_MASK_DIR = "example_masks_MIP_images"
OUTPUT_RESULTS_DIR = "output"

CONDITIONS = ["condition1", "condition2"]

SEGMENT_CHANNEL = "ch00"
RATIO_NUM = "ch01"
RATIO_DEN = "ch00"

USE_GPU = True
DIAMETER = None
```

Key settings:

| Setting | Purpose |
|---|---|
| `ROOT_DIR` | Folder containing the input MIP images |
| `OUTPUT_MASK_DIR` | Folder where Cellpose masks and overlays are saved |
| `OUTPUT_RESULTS_DIR` | Folder where CSV files and plots are saved |
| `CONDITIONS` | Condition folders to analyze |
| `SEGMENT_CHANNEL` | Channel used for Cellpose segmentation |
| `RATIO_NUM` | Numerator channel for ratio calculation |
| `RATIO_DEN` | Denominator channel for ratio calculation |
| `USE_GPU` | Use GPU for Cellpose if available |
| `DIAMETER` | Cell diameter for Cellpose; `None` allows automatic estimation |

---

## How to run

1. Place your MIP images inside `example_MIP_images/`.
2. Make sure the folder structure follows the required format.
3. Open:

```bash
jupyter notebook "Cellpose_condition1_condition2_ratio_analysis_v2.ipynb"
```

4. Check the configuration cell.
5. Run all notebook cells in order.
6. Inspect segmentation overlays in `example_masks_MIP_images/`.
7. Use the CSV tables and plots in `output/` for analysis.

---

## Important notes

- This workflow does not generate MIPs.
- This workflow does not analyze Z-stacks.
- This workflow does not use pH folders.
- Input images must already be named `ch00_MIP.tif` and `ch01_MIP.tif`.
- The only expected biological/experimental groups are `condition1` and `condition2`.
- Segmentation quality must be checked using the overlay PNG files before trusting the final ratios.

---

## Previous workflow changes

The older notebook expected raw Z-stack images and used condition/pH/view/channel grouping.

This corrected notebook removes:

- Z-stack parsing
- automatic MIP generation
- pH-based grouping
- filename metadata parsing

The corrected notebook keeps:

- Cellpose segmentation
- mask and overlay generation
- single-cell intensity extraction
- channel ratio calculation
- CSV and plot export
