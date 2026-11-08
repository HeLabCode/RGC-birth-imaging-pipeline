# FRET-Based Sensor Analysis Pipeline

## Overview

This pipeline measures per-cell FRET efficiency from aligned donor/Turquoise and FRET/acceptor-emission fluorescence images using a binary cell mask. It refines the binary mask into individual cell regions, estimates and subtracts background signal, applies the correction factors specified in the notebook, and exports per-cell measurements and pseudocolor images.

The included example demonstrates the analysis on one pair of 2D fluorescence images and an associated mask. It is not a workflow for generating a segmentation mask from raw images.

## Repository organization

| File or folder | Contents |
| --- | --- |
| [`FRET_analysis.ipynb`](FRET_analysis.ipynb) | Main analysis notebook: configuration, mask refinement, background correction, FRET calculation, and visualization |
| [`4.1 FRET analysis functions/`](4.1%20FRET%20analysis%20functions/) | Python helper module [`fret_utils.py`](4.1%20FRET%20analysis%20functions/fret_utils.py) used by the notebook |
| [`4.2 Example images and mask/`](4.2%20Example%20images%20and%20mask/) | Donor image, FRET-channel image, and binary mask |
| [`4.3 FRET analysis output/`](4.3%20FRET%20analysis%20output/) | Included example results and default destination for generated outputs |

## Requirements

The notebook uses Python, Jupyter, NumPy, pandas, tifffile, Matplotlib, SciPy, and scikit-image.

```bash
pip install jupyter numpy pandas tifffile matplotlib scipy scikit-image
```

Run the notebook from the **`4. FRET based sensor analysis pipeline`** folder so that its relative input and helper-module paths resolve correctly:

```bash
jupyter lab FRET_analysis.ipynb
```

## Example input data

The example contains these three 300 × 300-pixel, 8-bit TIFF files:

| Input file | Role |
| --- | --- |
| [`donor.tif`](4.2%20Example%20images%20and%20mask/donor.tif) | Donor/Turquoise fluorescence image (`ch1` in notebook descriptions) |
| [`acceptor.tif`](4.2%20Example%20images%20and%20mask/acceptor.tif) | FRET/acceptor-emission fluorescence image (`ch3` in notebook descriptions) |
| [`binary_mask.tif`](4.2%20Example%20images%20and%20mask/binary_mask.tif) | Binary cell-region mask used for downstream instance separation |

The images and mask are the same size. The notebook requires donor and FRET images to have identical shapes; its mask loader also **requires** the mask shape to match and does not automatically resize it. The mask can originate from a neural-network or another segmentation workflow, but its generation is not included in this pipeline.

## Run the example

1. Open [`FRET_analysis.ipynb`](FRET_analysis.ipynb) from the pipeline folder.
2. Use the input paths already configured in the **User settings** cell. These refer to the three TIFFs in `4.2 Example images and mask/`.
3. Run the notebook cells in order. The notebook loads the input images and mask, refines the mask into cell labels, and saves a segmentation preview.
4. Inspect `segmentation_preview.tif` to assess how well connected regions have been separated into cells.
5. Continue through background correction, FRET-efficiency calculation, CSV export, and pseudocolor visualization.
6. Inspect the outputs in `4.3 FRET analysis output/`.

The example outputs are already included in the archive. Running the notebook again with its default settings writes to the same filenames and can replace those files.

## Analysis workflow

### 1. Load images and mask

The notebook loads donor and FRET images as intensity arrays and converts the binary mask to a nonzero/zero cell mask. Input dimensions are checked before subsequent calculations.

### 2. Separate cells from the binary mask

The helper functions apply distance-transform/watershed-based refinement to separate connected mask regions into labeled cell instances. Small segmented regions are filtered. The resulting cell outlines and labels are shown over the donor image in a segmentation preview.

### 3. Estimate and subtract background

With background correction enabled, background pixels are selected from non-cell regions away from cell boundaries. A background estimate is calculated separately for the donor and FRET channels and subtracted from the respective images.

### 4. Compute corrected FRET efficiency

The notebook uses the following calculation on the background-corrected intensities:

```text
I_Turq     = I_Turq_corr × C
I_FRET     = I_FRET_corr × D
FRET_corr  = (1 − A) × I_FRET − B × I_Turq
D_lost     = FRET_corr × QYd_QYa_ratio × Sd_Sa_ratio × Gd_Ga_ratio
E_FRET     = D_lost / (D_lost + I_Turq)
```

This is the **calculated, correction-dependent FRET-efficiency quantity defined by the code**. Its interpretation depends on the correction/calibration constants used; the included example does not independently establish the calibration of another experimental setup. Nonfinite results are treated as missing values, and the helper function clips finite pixel-wise values to its default range of −10 to 10.

For each labeled cell, the script slightly shrinks the region and calculates the mean pixel-wise FRET value within it. The CSV includes the cell identifier, centroid, area, mean FRET value, and calculation constants.

### 5. Generate visualizations

The notebook colors cell regions by their mean FRET value and exports a separate labeled color scale. The pseudocolor display range is a visualization setting, not a filter on the values written to the CSV.

## Parameters used in the example

| Parameter | Notebook value | Purpose |
| --- | --- | --- |
| `WATERSHED_MIN_DISTANCE` | `8` | Watershed peak separation |
| `MIN_CELL_SIZE` | `40` | Minimum segmented-region size |
| `PEAK_THRESH` | `0.15` | Segmentation peak threshold |
| `SHRINK_FACTOR` | `0.95` | Shrinkage of per-cell measurement regions |
| `DO_BACKGROUND_SUBTRACTION` | `True` | Enable background correction |
| `BACKGROUND_KEEP_FRACTION` | `0.95` | Fraction of more distant non-cell pixels retained for background selection |
| `BACKGROUND_PERCENTILE` | `50` | Percentile used to estimate background |
| `A`, `B`, `C`, `D` | `0.016`, `0.11`, `1.0`, `1.0` | Correction and intensity-scaling factors |
| `QYd_QYa_ratio` | `1.474` | Quantum-yield ratio factor |
| `Sd_Sa_ratio`, `Gd_Ga_ratio` | `1.0`, `1.0` | Spectral-sensitivity and detection-gain ratio factors |
| `EFRET_VMIN`, `EFRET_VMAX` | `0.7`, `1.0` | Pseudocolor display limits |
| `EFRET_CMAP` | `coolwarm` | Pseudocolor map |

These values are recorded from the supplied notebook. Different acquisition settings or sensors can require different correction factors; the notebook does not estimate those factors from the example files.

## Outputs

The default output directory is [`4.3 FRET analysis output/`](4.3%20FRET%20analysis%20output/). It contains the following example files:

| Output file | Description |
| --- | --- |
| [`cell_FRET_efficiency.csv`](4.3%20FRET%20analysis%20output/cell_FRET_efficiency.csv) | Per-cell FRET values, positions, areas, and correction parameters (38 rows in the included example) |
| [`segmentation_preview.tif`](4.3%20FRET%20analysis%20output/segmentation_preview.tif) | Donor image with labeled segmentation outlines |
| [`EFRET_per_cell_pseudocolor.tif`](4.3%20FRET%20analysis%20output/EFRET_per_cell_pseudocolor.tif) | Per-cell pseudocolor visualization |
| [`EFRET_color_scale_bar.tif`](4.3%20FRET%20analysis%20output/EFRET_color_scale_bar.tif) | Labeled pseudocolor scale |

## Notes on reproducibility

- Use aligned fluorescence channels and a corresponding mask with matching dimensions.
- Inspect watershed segmentation before interpreting individual cell measurements.
- Keep correction constants and visualization ranges consistent when comparing analyses performed with the same acquisition setup.
- The included TIFFs are an analysis example. They do not, by themselves, document the acquisition history or calibration of a complete experimental dataset.
- Results depend on segmentation, background selection, and correction constants. The notebook and example outputs are supplied, but this README does not claim independent end-to-end execution validation.
