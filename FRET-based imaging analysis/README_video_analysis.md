# Video Analysis Pipeline

## Overview
This notebook analyzes time-lapse microscopy videos using manually defined cell masks, drift correction, centroid extraction, and cell tracking.

The workflow loads selected video frames and corresponding segmentation masks, corrects global image drift using manually selected anchor points, tracks cells across frames, and generates quality-control videos and trajectory visualizations.

This notebook is marked as it has not been completed yet

---

## Features
- Load multi-frame TIFF image sequences
- Match video frames with manually generated mask files
- Preview raw videos
- Manual anchor-point collection for drift correction
- Global drift interpolation and correction
- Drift correction of image stacks and mask stacks
- Overlay masks on raw or corrected videos
- Extract cell centroids from labeled masks
- Track cells across frames using TrackPy
- Merge nearby duplicated tracks
- Remove short or blinking tracks
- Visualize centroid projections
- Display tracking overlays in Jupyter
- Save overlay and tracking videos

---

## Requirements

Install the required packages:

```bash
pip install numpy pandas matplotlib tifffile scikit-image scipy opencv-python natsort trackpy imagecodecs
```

---

## Input Data

The notebook expects:

- A folder containing time-lapse image frames, usually TIFF files
- A folder containing manually generated segmentation masks, usually `.npy` files
- Matching frame indices between image files and mask files

Example expected data:
- raw image frames: `*.tif`
- mask files: `*_regiongrow_labels_T*.npy`

The image stack and mask stack must refer to the same selected time points.

---

## Workflow

### 1. Load Videos
The notebook loads selected TIFF frames from the experiment folder and organizes them by channel.

It also detects which time points have corresponding mask files, then keeps only frames available across all required channels.

---

### 2. Preview Video
Raw image stacks can be played using OpenCV to check:

- image quality
- motion
- channel visibility
- frame ordering
- obvious acquisition artifacts

---

### 3. Manual Anchor Point Selection
Anchor points are selected manually across the video.

These points are used to estimate global motion of the field of view over time.

---

### 4. Global Drift Correction
The selected anchor positions are interpolated across frames.

The resulting shift vectors are used to correct:

- raw image stack
- labeled mask stack

Outputs can include:

- drift vector plot
- `shift_vectors.npy`
- `drift_corrected_stack.tif`

---

### 5. Load Manual Masks
The notebook loads manually defined segmentation masks from `.npy` files.

Supported mask formats are converted into labeled mask images so each cell has a unique label.

---

### 6. Mask and Image Overlay
The notebook can display or save overlays showing segmentation masks on top of the raw or drift-corrected video.

Example outputs:

- `overlay.mp4`
- `overlay_drift_corrected.mp4`

These videos are used for quality control before tracking.

---

### 7. Centroid Extraction
Centroids are extracted from labeled masks for every frame.

Each detected object is converted into a coordinate table containing:

- frame index
- x coordinate
- y coordinate
- object identity

---

### 8. Cell Tracking
Cell centroids are linked across frames using TrackPy.

The pipeline can then:

- assign persistent track IDs
- merge nearby duplicated tracks
- remove unstable blinking tracks
- keep longer, more reliable tracks

---

### 9. Tracking Visualization
The notebook provides several visualization tools:

- 2D centroid projection
- tracking overlay playback
- tracking overlay with masks
- single-cell trajectory playback
- saved tracking overlay videos
- saved trajectory trail videos

These outputs help verify whether tracks are biologically plausible and technically stable.

---

## Output

Typical outputs include:

- drift-corrected image stack
- drift vectors
- mask overlay videos
- tracking overlay videos
- trajectory visualizations
- cleaned tracking table
- centroid projection plots

---

## Key Parameters

Adjust these before reuse:

- experiment folder path
- mask folder path
- image extension
- channel name
- video playback FPS
- anchor selection interval
- TrackPy search range
- track merging distance
- minimum track length
- maximum allowed frame gap
- output video FPS

---

## Notes

- This notebook contains hard-coded local paths and should be edited before running on a new system.
- Manual anchor placement strongly affects drift correction quality.
- Track quality should be checked visually before downstream analysis.
- Mask quality determines centroid and tracking accuracy.
- Save intermediate outputs so drift correction and tracking can be inspected independently.
