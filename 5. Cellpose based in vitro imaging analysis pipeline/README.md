# Cellpose Segmentation and Single-Cell Ratio Analysis

## Overview

This pipeline segments cells in two-channel maximum-intensity projection (MIP) fluorescence images using Cellpose, extracts background-corrected single-cell fluorescence intensities, calculates channel ratios, and produces tabular results and distribution plots.

The supplied example is organized into two conditions (`condition1` and `condition2`), each containing three `pH_n*` groups with three fields of view (`view1`–`view3`) per group. The folder labels describe the dataset organization; the notebook does not specify biological conditions beyond those labels.

## Repository organization

| File or directory | Contents |
| --- | --- |
| [`Cellpose_segmentation&analysis.ipynb`](Cellpose_segmentation%26analysis.ipynb) | Main notebook for segmentation, intensity extraction, quality filtering, and plots |
| [`5.1 Example images/`](5.1%20Example%20images/) | 36 channel TIFFs (`ch00_MIP.tif` and `ch01_MIP.tif` for 18 fields), plus stored mask files in additional `_mask` directories |
| [`5.2 Example images mask/`](5.2%20Example%20images%20mask/) | Separate copies of 18 segmentation label arrays and 18 overlay images |
| [`5.3 Example analysis output/`](5.3%20Example%20analysis%20output/) | Example CSV outputs and per-condition histogram plots |

The notebook's default configuration **writes newly generated masks into `5.1 Example images/condition1_mask/` and `condition2_mask/`**, rather than into `5.2 Example images mask/`. The separate `5.2` folder is provided as a reference set. Both directory locations are present in the supplied archive.

## Requirements

Run the notebook using Python with Jupyter and the following packages:

```bash
pip install numpy pandas matplotlib tifffile scikit-image cellpose jupyterlab
```

Cellpose interfaces can differ between versions. The included notebook uses `models.CellposeModel(gpu=True)` and calls `model.eval` with `diameter=None`, `normalize=True`, and `invert=False`. GPU execution is explicitly requested in the notebook; no automatic CPU fallback is implemented in the supplied code.

## Input data

Each field of view contains two spatially aligned 512 × 512 pixel, 8-bit TIFF MIPs:

```text
5.1 Example images/
├── condition1/
│   ├── pH_n1/
│   │   ├── view1/
│   │   │   ├── ch00_MIP.tif
│   │   │   └── ch01_MIP.tif
│   │   ├── view2/
│   │   └── view3/
│   ├── pH_n2/
│   └── pH_n3/
└── condition2/
    ├── pH_n1/
    ├── pH_n2/
    └── pH_n3/
```

Each `pH_n*` group contains `view1`, `view2`, and `view3`, with the same paired channel filenames. The images are already MIPs: the notebook does not perform Z-stack projection or registration.

## Running the example

1. Extract the ZIP, preserving the folder structure.
2. Open `Cellpose_segmentation&analysis.ipynb` from inside the pipeline directory so relative paths resolve correctly.
3. Review the configuration cell. Its default input path is `./5.1 Example images` and output path is `./5.3 Example analysis output`.
4. Run the image-discovery cell to identify paired `ch00` and `ch01` TIFF files under each condition.
5. Run Cellpose segmentation, or use the included labeled masks if running the downstream analysis separately. The notebook's segmentation cell writes masks and overlay PNGs into the `_mask` folders under `5.1 Example images`.
6. Run intensity extraction and filtering to generate CSV tables.
7. Run the final plotting cell to save per-condition intensity histograms.

This is a workflow guide based on notebook inspection, not a claim of independent end-to-end execution in a fresh environment. Segmentation results can vary with the Cellpose version and model.

## Analysis workflow

### 1. Cell segmentation

Cellpose is applied to the `ch00` MIP for each field. The notebook uses automatic diameter estimation (`diameter=None`) and normalization, then saves labeled masks as `ch00_MIP_labels.npy` and overlays as `ch00_MIP_overlay.png`.

### 2. Background correction

For each image channel, the median intensity in pixels outside segmented cells (label 0) is subtracted from cell pixel intensities. The mean corrected intensity is calculated separately for `ch00` and `ch01`.

### 3. Cell filtering and ratio calculation

By default, the ratio is:

```text
mean background-corrected ch01 / mean background-corrected ch00
```

The code excludes cells when the corrected denominator signal is below three times the estimated background noise, more than 1% of pixels in either channel are at or above intensity 255, or the denominator is nonpositive. Background noise is initially estimated using the median absolute deviation with fallback estimates where needed.

Default parameters from the configuration cell:

| Parameter | Value |
| --- | --- |
| `conditions` | `condition1`, `condition2` |
| `segmentation_channel` | `ch00` |
| `numerator_channel` | `ch01` |
| `denominator_channel` | `ch00` |
| `saturation_value` | `255` |
| `max_saturated_fraction` | `0.01` |
| `min_denominator_signal_to_noise` | `3` |
| Cellpose `gpu` | `True` |

### 4. Output plots

For each condition, the notebook plots distributions of background-corrected `ch00` and `ch01` mean intensities before and after cell filtering. The histograms are **intensity histograms**, not ratio histograms.

## Included output files

All the following example results are in [`5.3 Example analysis output/`](5.3%20Example%20analysis%20output/):

| Filename | Description |
| --- | --- |
| `single_cell_raw_before_filtering.csv` | Per-cell background-corrected channel means before quality filtering |
| `single_cell_ratios.csv` | Retained cells, corrected channel means, computed fluorescence ratio, and cell area |
| `filtering_summary.csv` | Per-field counts of total and excluded cells |
| `condition1_histograms.png`, `condition1_histograms.pdf` | Before/after filtering intensity plots for condition 1 |
| `condition2_histograms.png`, `condition2_histograms.pdf` | Before/after filtering intensity plots for condition 2 |

## Notes on interpretation

- `condition1`, `condition2`, and `pH_n1`–`pH_n3` are folder identifiers; no specific experimental treatment or pH value is asserted here.
- Cell masks should be inspected before interpreting per-cell measurements.
- The code analyzes images that are already aligned MIPs, and the computed channel ratio depends on the configured numerator and denominator.
- The existing notebook uses GPU-enabled Cellpose and has no built-in installation or version-locking procedure.
- Rerunning the notebook can overwrite existing masks and output files in its configured destinations.

## Reference

Stringer C, Wang T, Michaelos M, Pachitariu M. Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*. 2021;18:100–106.
