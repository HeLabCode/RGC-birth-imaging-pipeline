# pHLemon-rev FRET Sensor Video (Figure 5b)

## Overview

This notebook generates a video of pHLemon-rev fluorescence imaging, combining donor and acceptor images, per-cell FRET pseudocolor maps, and the corresponding mean FRET time course.

The supplied files include paired fluorescence images, cell masks, and a table of previously calculated per-cell FRET measurements. This notebook visualizes these prepared results; it does not perform neural-network segmentation or recalculate FRET efficiency.

This notebook uses precomputed segmentation masks; it does not run a segmentation model.

The available files have nonconsecutive time-point identifiers. The visualization uses the supplied frames and FRET values; missing time points are not additional measurements.

---

## Features

- Visualization of donor and acceptor fluorescence images
- Per-cell FRET pseudocolor maps using supplied segmentation masks
- Display of the mean FRET time course and cell-to-cell variability
- Alignment of the displayed images with their corresponding FRET measurements
- Video export in MP4 format
- CSV export of the mean FRET time course

---

## Repository Structure

### `pHLr_FRET_video.ipynb`
Main notebook for loading the prepared data, previewing the visualization, and exporting the video.

### `functions.py`
Helper functions for image loading, FRET visualization, plotting, and video export.

### `Input analysis results/`
Prepared dataset used to reproduce the representative visualization:

```text
Input analysis results/
├── donor/
│   └── pHLr_C001T*.tif
├── acceptor/
│   └── pHLr_C002T*.tif
├── cell_masks/
│   └── pHLr_labels_T*.tif
└── cell_FRET.csv
```

The `T` identifier is used to associate donor, acceptor, and mask files. Not all integer time-point identifiers are present in the supplied files.

### `Output exported video/`
Folder containing the generated visualization outputs:

```text
Output exported video/
├── mean_FRET_per_view.csv
└── pHLr_analysis_video_results.mp4
```

---

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scipy tifffile ipywidgets ipython imageio imageio-ffmpeg
```

Run `pHLr_FRET_video.ipynb` with this folder as the working directory, keeping `functions.py` beside the notebook. Execute its cells in order; the export cell writes the MP4 and CSV into `Output exported video/`.

---

## Input Data

The notebook uses:

- **Donor images:** TIFF images from the donor fluorescence channel (`C001`)
- **Acceptor images:** corresponding TIFF images from the acceptor fluorescence channel (`C002`)
- **Cell masks:** previously generated neural-network masks aligned with the fluorescence images
- **FRET measurements:** `cell_FRET.csv`, containing previously calculated per-cell FRET values

These are the input images used by this visualization notebook. The notebook does not regenerate the masks or the FRET measurements.

---

## Run the Included Example

1. Open `pHLr_FRET_video.ipynb` from this folder.
2. Install the listed dependencies and run the configuration cell using `Input analysis results/`.
3. Preview the visualization using the notebook controls.
4. Run the export cell and inspect `Output exported video/pHLr_analysis_video_results.mp4` and `Output exported video/mean_FRET_per_view.csv`.

The included example uses prepared results; the notebook does not recompute FRET values from the original acquisition.

---

## Workflow

### 1. Image and FRET Data Loading
- Loads the available donor and acceptor TIFF images
- Retrieves the corresponding cell masks and previously calculated FRET measurements
- Matches the available image frames using their time-point identifiers

### 2. FRET Visualization
- Displays the donor and acceptor images alongside per-cell FRET pseudocolor maps
- Shows the corresponding mean FRET time course and cell-level variability
- Associates each displayed frame with its position in the plotted time course

### 3. Video Generation
- Generates a video from the available frames
- Displays the prepared measurements across the selected time points
- Exports the combined visualization as an MP4 file

### 4. Results Export
- Saves the video in the designated output folder
- Exports a CSV containing the mean FRET time course

---

## Output

| File | Description |
| --- | --- |
| `pHLr_analysis_video_results.mp4` | Video combining fluorescence images, per-cell FRET maps, and the mean FRET time course |
| `mean_FRET_per_view.csv` | Summary of the mean FRET time course and cell-level variability |

Both files are stored in `Output exported video/`.

---

## Parameters to Adjust

- `DATA_DIR` — input folder (`Input analysis results/`); the notebook uses a path relative to its working directory
- `OUTPUT_DIR` — output folder (`Output exported video/`)
- `sample_name` — prefix used to match the input files (`pHLr`)
- `VIDEO_FRAME_INDICES` — frames selected for video export (`None` uses all available frames)
- `VisualizationSettings()` — display settings passed to the visualization functions

---

## Notes

- The provided masks and FRET values are inputs to this visualization, not outputs of this notebook.
- The available time-point identifiers are nonconsecutive; do not interpret displayed intermediate positions as newly acquired measurements.
- The plotted cell-to-cell variability is not a measure of variation across biological replicates.
- Run the notebook from its own folder so that the relative input and output paths resolve correctly.
