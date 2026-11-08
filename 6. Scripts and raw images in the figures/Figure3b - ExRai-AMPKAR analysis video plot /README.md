# ExRai-AMPKAR Z-Stack Videos and Intensity Heatmaps

## Overview

This notebook generates Z-stack videos and per-cell intensity heatmaps from ExRai-AMPKAR two-photon imaging datasets acquired with alternating 800 nm and 920 nm excitation.

It combines the original image stacks with previously generated cell masks, background-subtracted fluorescence intensities, and final excitation ratios. The resulting videos show how the same tracked cells appear across Z-planes, with their maximum-intensity planes highlighted in each excitation channel.

This notebook is a **visualization step** that reads previously saved segmentation, tracking, background-subtraction, and ratio results. It does not recompute those measurements.

---

## Features

- Paired 800 nm and 920 nm Z-stack visualization
- Pseudocolor projection of per-cell 920/800 excitation ratios
- Consistent ratio color scale across time points
- Background-subtracted intensity heatmaps for both channels
- Cell matching using the final S7 correspondence table
- Identification of each cell's maximum-intensity Z-plane in each channel
- Colored peak-plane outlines and heatmap markers
- AVI video export and CSV tables with peak-plane information

---

## Repository Structure

### `ExRai_AMPKAR_heatmaps_video.ipynb`
Main notebook for generating the videos and intensity heatmaps.

### `Input analysis results/`
Contains the time-point folders (`0min/`, `1min/`, `3min/`, `6min/`, `8min/`) with TIFF images and saved analysis results. Each time-point folder follows this structure:

```text
Input analysis results/
└── 0min/
    ├── 800_ch2_Z001.tif
    ├── 800_ch2_Z002.tif
    ├── ...
    ├── 920_ch2_Z001.tif
    ├── 920_ch2_Z002.tif
    ├── ...
    └── results/
        ├── S2_tracks_800/
        │   └── 800_track0001_maskstack.tif
        ├── S2_tracks_920/
        │   └── 920_track0001_maskstack.tif
        ├── S5_backg_subtraction_800/
        │   └── 800_S5_means_bgsub.csv
        ├── S5_backg_subtraction_920/
        │   └── 920_S5_means_bgsub.csv
        └── S7_final_output/
            └── S7_manual_ratio_results.csv
```

The same structure is used for the other time points. Track numbers are illustrative; mask-stack names must match the cell IDs in the S7 results table.

### `Output folder/`
Created automatically by the notebook and contains all generated AVI videos and peak-alignment CSV tables.

---

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib tifffile opencv-python pillow ipython
```

Open `ExRai_AMPKAR_heatmaps_video.ipynb` with this folder as the working directory, and run the cells in order. The default configuration processes five time-point folders and writes AVI videos and CSV tables into `Output folder/`.

---

## Input Data

For each time point, the notebook requires:

- **Raw Z-stack images:** `800_ch2_Z*.tif` and `920_ch2_Z*.tif`, from the same field of view
- **S2 masks:** tracked cell mask stacks for both excitation channels
- **S5 intensity tables:** `800_S5_means_bgsub.csv` and `920_S5_means_bgsub.csv`, containing background-subtracted intensities by cell and Z-plane
- **S7 results:** `S7_manual_ratio_results.csv`, containing matched `track_800`/`track_920` IDs and `ratio_920_over_800` values

The notebook sets `BASE_DIR` to `Path.cwd() / "Input analysis results"`. Run it from its own folder to resolve this default correctly.

---

## Run the Included Example

1. Open `ExRai_AMPKAR_heatmaps_video.ipynb` with this folder as the working directory.
2. Install the listed dependencies.
3. Run the notebook in order using the existing `Input analysis results/` folders.
4. Inspect the generated videos and peak-alignment CSV files in `Output folder/`.

The input folders contain previously processed S2/S5/S7 results. Running this notebook creates visualizations rather than recalculating the underlying cell measurements.

---

## Workflow

### 1. Load Images and Analysis Results
- Load 800 nm and 920 nm TIFF planes in Z order
- Read final S7 cell ratios and tracked cell masks
- Match the two excitation stacks for video display

### 2. Generate Z-Stack Videos
- Display Ex800, Ex920, and the per-cell pseudocolor ratio in three panels
- Use a fixed ratio range and colormap across time points
- Save one AVI video for each time point

### 3. Match Cells and Identify Peak Planes
- Use the S7 table to match Ex800 and Ex920 cell IDs
- Read background-subtracted intensities from S5
- Identify the maximum-intensity Z-plane independently for each channel

### 4. Generate Intensity Heatmaps
- Arrange matched cells in the same column order for both channels
- Display intensity across Z using channel-specific grayscale ranges
- Highlight each cell's peak plane with a colored rectangle
- Draw the corresponding colored outline on the excitation image

### 5. Export Results
- Save the videos with heatmaps below the image panels
- Export a CSV containing cell correspondences, peak planes, and outline colors

---

## Output

The notebook saves generated files in `Output folder/`, created relative to its working directory. Input data in `Input analysis results/` remain separate:

| File | Description |
| --- | --- |
| `ExRai_<timepoint>_Zstack.avi` | Three-panel Z-stack video with excitation images and pseudocolor ratios |
| `ExRai_<timepoint>_Zstack_video.avi` | Z-stack video with peak-plane outlines and intensity heatmaps |
| `ExRai_<timepoint>_peak_alignment.csv` | Matched cell IDs, peak Z-planes, and displayed outline colors |

The three files are produced separately for each time point listed in `TIMEPOINTS`.

---

## Parameters to Adjust

At the beginning of the notebook:

- `BASE_DIR` — location of the input time-point folders (`Input analysis results/`)
- `OUTPUT_DIR` — location of the generated outputs (`Output folder/`)
- `TIMEPOINTS` — time points to process
- `EX800_RANGE`, `EX920_RANGE` — brightness limits for the original image panels
- `RATIO_RANGE`, `COLOR_STOPS` — pseudocolor scale for the ratio panel
- `FPS` — video playback speed
- `SHOW_NOTEBOOK_PREVIEW` — enable or disable interactive previews

Heatmap dimensions, intensity display limits, spacing, and peak-outline thickness can be adjusted in the corresponding visualization cells.

---

## Notes

- The required S2, S5, and S7 results are already included in the supplied input folders. This notebook reads those results; it does not run upstream segmentation or ratio analysis.
- The S5 file lookup expects **exactly one matching CSV per excitation channel** within each time point. If multiple copies match, the notebook raises an error; keep only one matching file or update the S5 lookup path.
- The ratio display range in this notebook defaults to **0.2–0.5**. This is a visualization setting and may differ from the range used to generate the earlier S7 projection images.
- The Ex800 and Ex920 heatmaps have separate, fixed intensity display scales. These affect contrast only, not the underlying S5 measurements or selected peak planes.
- Confirm cell correspondence, mask alignment, and Z-plane ordering before comparing time points.
