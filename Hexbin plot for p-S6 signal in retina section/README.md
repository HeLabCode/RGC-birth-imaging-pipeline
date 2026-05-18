# Hexbin Ratio Map Generation

## Overview
This project generates spatial hexbin ratio maps from paired fluorescence images (p-S6 and RBPMS).  
The output visualizes relative p-S6 signal across RBPMS-positive cells while preserving retinal structure.

The workflow is designed for retinal imaging datasets where spatial context is important.

---

## Features
- Segmentation of RBPMS-positive cells
- Tissue mask generation for anatomical context
- Background estimation and subtraction
- Hexbin-based spatial aggregation
- Ratio calculation: mean(p-S6) / mean(RBPMS)
- Export of vector figures (PDF, SVG)

---

## Repository Structure

### `example_images/`
Example input images used to test the hexbin map generation.

- Includes paired p-S6 and RBPMS images
- Images are cropped, aligned, and ready for analysis

### `output_hexbin_plot/`
Folder containing the generated hexbin map outputs.

### `Hexbin_plot_code.ipynb`
Main notebook used to run the hexbin analysis and generate the maps.


---
## Requirements
Install required packages:

```bash
pip install tifffile numpy matplotlib scipy
```

---

## Input Data
- Cropped TIFF images (from Fiji/ImageJ)
- Two aligned channels:
  - p-S6 image
  - RBPMS image

Images must:
- Be restricted to retinal region (e.g., using *Clear Outside* in Fiji)
- Have identical dimensions
- Be spatially aligned

---

## Workflow

### 1. Load Images
Import TIFF files as single-channel 2D arrays.

### 2. Segmentation
- RBPMS mask:
  - Gaussian smoothing
  - Percentile thresholding
  - Morphological cleanup
  - Connected component filtering

- Tissue mask:
  - Generated from p-S6 image
  - Excludes RBPMS-positive regions

### 3. Background Subtraction
- Estimate background using tissue mask
- Subtract median background per channel
- Clip negative values to zero

### 4. Hexbin Aggregation
- Apply hexagonal binning to RBPMS-positive pixels
- Compute per-bin:
  - mean p-S6
  - mean RBPMS

### 5. Ratio Map
- Calculate ratio per bin:
  - p-S6 / RBPMS
- Exclude bins with zero RBPMS
- Overlay on tissue mask for spatial context

---

## Output
- Hexbin ratio map visualization
- Export formats:
  - PDF
  - SVG

---

## Parameters to Adjust
When applying to new datasets:
- Segmentation thresholds (percentiles)
- Minimum object size for masks
- Hexbin `gridsize`
- Display intensity range

Keep visualization settings consistent for cross-sample comparison.

---

## Notes
- Always visually inspect masks before proceeding
- Ensure proper alignment of input channels
- Maintain consistent plotting settings across experiments
