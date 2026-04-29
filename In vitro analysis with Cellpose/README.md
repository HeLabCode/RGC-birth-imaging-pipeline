# Cellpose Segmentation and Single-Cell Ratio Analysis

A Python/Jupyter workflow for automated segmentation and fluorescence ratio analysis of in vitro HEK cell images using [Cellpose](https://www.cellpose.org/).

The notebook takes raw multi-channel Z-stack TIFF images, generates maximum-intensity projections, segments cells with Cellpose, extracts single-cell fluorescence intensities, filters low-quality measurements, and saves clean CSV outputs for downstream analysis.

## What this workflow does

1. **Builds maximum-intensity projections** from Z-stack TIFF images.
2. **Segments cells** on a selected reference channel using Cellpose.
3. **Saves segmentation masks** and overlay images for visual inspection.
4. **Extracts per-cell fluorescence intensities** from two channels.
5. **Applies quality-control filters** to remove dim, saturated, or invalid cells.
6. **Calculates single-cell fluorescence ratios**, such as `ch01 / ch00`.
7. **Exports CSV tables and QC plots** for further analysis.


## Repository contents

```text
.
├── Cellpose_segmentation&analysis.ipynb   # Main analysis notebook
└── README.md                              # Project documentation
```

## Requirements

Install the required Python packages with:

```bash
pip install numpy pandas matplotlib seaborn tifffile scikit-image scipy opencv-python cellpose
```

Recommended environment:

- Python 3.9+
- Jupyter Notebook or JupyterLab
- GPU support is optional but recommended for faster Cellpose segmentation

## Input data format

The workflow expects raw TIFF images named with condition, pH, field of view, Z-slice, and channel information.

Example filename pattern:

```text
sample_TA_7.29_v1_z3_ch00.tif
```

Expected naming fields:

| Field | Example | Meaning |
|---|---:|---|
| Condition | `TA`, `WT` | Experimental condition |
| pH | `7.29` | pH value |
| View | `v1` | Field of view |
| Z-slice | `z3` | Z-stack index |
| Channel | `ch00`, `ch01` | Imaging channel |

## Workflow

### 1. Generate MIP images

The notebook groups raw TIFF files by condition, pH, view, and channel. It then stacks the Z-slices and saves one maximum-intensity projection per group.

Output structure:

```text
Raw_images/
└── CONDITION/
    └── pH_VALUE/
        └── VIEW/
            ├── ch00_MIP.tif
            ├── ch01_MIP.tif
            └── ch02_MIP.tif
```

### 2. Run Cellpose segmentation

Cellpose is run on a selected channel, for example `ch00` or `ch01`. The notebook saves:

- Label masks as `.npy` files
- Overlay images as `.png` files

Example outputs:

```text
MIP_images/
├── mask_TA/
│   └── pH_7.29/
│       └── v1/
│           ├── ch00_labels.npy
│           └── ch00_overlay.png
```

### 3. Extract single-cell ratios

For each segmented cell, the notebook calculates background-corrected mean fluorescence intensities and computes:

```text
ratio = ch01_mean / ch00_mean
```

Cells are filtered out when they are:

- Too dim in the denominator channel
- Saturated in either channel
- Invalid because the denominator intensity is zero or negative

## Output files

The final results are saved in the `results` folder.

```text
results/
├── single_cell_ratios.csv
├── single_cell_raw_before_filtering.csv
├── filtering_summary.csv
├── ATEAM_overlay_histograms.png
└── DRUG_overlay_histograms.png
```

### Main CSV outputs

| File | Description |
|---|---|
| `single_cell_ratios.csv` | Filtered single-cell fluorescence ratios and metadata |
| `single_cell_raw_before_filtering.csv` | Raw per-cell intensity measurements before filtering |
| `filtering_summary.csv` | Per-image summary of retained and removed cells |

## Configuration

Before running the notebook, update these paths and parameters to match your dataset:

```python
ROOT = Path("/path/to/raw/images")
ROOT_DIR = "/path/to/MIP_images"
CHANNEL_TAG = "ch00"
MASK_CHANNEL = "ch01"
USE_GPU = True
DIAMETER = None
```

Key settings:

| Setting | Purpose |
|---|---|
| `ROOT` | Folder containing raw Z-stack TIFF files |
| `ROOT_DIR` | Folder containing generated MIP images |
| `CHANNEL_TAG` | Channel used for Cellpose segmentation |
| `MASK_CHANNEL` | Channel whose segmentation mask is used for analysis |
| `USE_GPU` | Set to `True` for GPU-based Cellpose segmentation |
| `DIAMETER` | Cellpose cell diameter; use `None` for automatic estimation |

## How to run

1. Clone this repository.
2. Install the required packages.
3. Place your raw TIFF images in the expected input folder.
4. Open the notebook:

```bash
jupyter notebook "Cellpose_segmentation&analysis.ipynb"
```

5. Update the input/output paths in the notebook.
6. Run the cells in order.
7. Check the `results` folder for CSV files and QC plots.

## Notes

- This workflow was designed for in vitro HEK cell fluorescence images.
- The notebook assumes two analysis channels, currently `ch00` and `ch01`.
- File naming must be consistent for automatic grouping to work.
- Segmentation quality should be checked using the saved overlay images before interpreting ratio results.

## License

No license has been specified yet. Add a license file if you plan to make this repository public.
