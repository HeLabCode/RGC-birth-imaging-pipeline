# p-S6/RBPMS Hexbin Visualization Pipeline

## Overview

This pipeline generates a spatial hexbin map of relative p-S6 fluorescence in RBPMS-positive regions of retinal sections. It uses matched p-S6 and RBPMS images, segments RBPMS-positive pixels, estimates background fluorescence, and visualizes the ratio **mean(p-S6) / mean(RBPMS)** across the retinal section.

The output preserves the outline of the surrounding retinal tissue for anatomical context. **The map is a visualization of local fluorescence ratios, not a count or percentage of p-S6-positive retinal ganglion cells (RGCs).**

## Repository organization

| Path | Description |
| --- | --- |
| [`Hexbin_plot_code.ipynb`](Hexbin_plot_code.ipynb) | Main notebook for mask generation, background subtraction, hexbin aggregation, and visualization |
| [`example_images/Pre_birth_p-S6.tif`](example_images/Pre_birth_p-S6.tif) | Example p-S6 fluorescence image |
| [`example_images/Pre_birth_RBPMS.tif`](example_images/Pre_birth_RBPMS.tif) | Matching example RBPMS fluorescence image |
| [`output_hexbin_plot/Hexbin_map.pdf`](output_hexbin_plot/Hexbin_map.pdf) | Example exported vector figure (PDF) |
| [`output_hexbin_plot/Hexbin_map.svg`](output_hexbin_plot/Hexbin_map.svg) | Example exported vector figure (SVG) |

## Example input data

| Attribute | Included example |
| --- | --- |
| Sample label in filenames | `Pre_birth` |
| Image types | p-S6 fluorescence; RBPMS fluorescence |
| Preparation | Retinal-section images, cropped/prepared for the analysis |
| Image dimensions | 2856 × 1955 pixels, with 3 stored samples per pixel |
| Data type | 8-bit unsigned integer TIFF |
| Notebook preprocessing | Maximum across the last image axis to produce one 2D intensity image per TIFF |
| Spatial relationship | Paired images must depict the same aligned field of view |

**Important:** The example TIFFs are stored as `Y × X × 3` arrays. The notebook converts each to a 2D intensity image using `np.max(..., axis=2)`; it does not load a Z-stack.

## Requirements

Install Python and Jupyter Notebook or JupyterLab, then install the analysis packages:

```bash
pip install numpy scipy matplotlib tifffile jupyterlab
```

## Running the example analysis

- [ ] Download or clone the repository, retaining the folders shown above.
- [ ] Open a terminal in **this module's folder** (the directory containing `Hexbin_plot_code.ipynb`).
- [ ] Start Jupyter with `jupyter lab` and open [`Hexbin_plot_code.ipynb`](Hexbin_plot_code.ipynb).
- [ ] Confirm that the input paths in the loading cell are `./example_images/Pre_birth_p-S6.tif` and `./example_images/Pre_birth_RBPMS.tif`.
- [ ] Run the notebook cells in order. Inspect the p-S6/RBPMS image previews and the RBPMS/tissue masks.
- [ ] Confirm that `output_hexbin_plot/` exists before the final export cell.
- [ ] Check the exported maps in [`output_hexbin_plot/`](output_hexbin_plot/): `Hexbin_map.pdf` and `Hexbin_map.svg`.

Running the notebook from its own folder matters because its input and output locations are **relative paths**. The notebook saves outputs using fixed filenames; rerunning it overwrites existing exported maps.

## Workflow

1. **Load the paired TIFFs.** Read the two example images and take the maximum across their final (three-sample) axis to create 2D p-S6 and RBPMS intensity images.
2. **Generate the RBPMS-positive mask.** Apply Gaussian smoothing, percentile thresholding, hole filling, morphological opening, and connected-component size filtering.
3. **Generate the tissue mask.** Segment the retinal context from the p-S6 image and remove pixels already selected by the RBPMS-positive mask.
4. **Estimate background.** Calculate the median intensity of each fluorescence image inside the tissue mask, and subtract each median from the corresponding image. The current code does **not** explicitly clip negative values to zero.
5. **Aggregate by hexagonal spatial bins.** Within RBPMS-positive pixels, calculate separately the mean background-subtracted p-S6 and RBPMS intensities for each hexagon.
6. **Calculate and visualize the ratio.** Divide the mean p-S6 value by the mean RBPMS value in bins with a nonzero RBPMS denominator. Overlay the ratio map on a low-intensity tissue-context layer and save PDF/SVG figures.

## Key parameters

The supplied notebook uses the following settings for the included example:

| Parameter | Example value | Purpose |
| --- | --- | --- |
| `percentile1` | `96.6` | RBPMS-positive mask threshold |
| `percentile2` | `50` | Tissue-mask threshold |
| `size1` | `2000` | Minimum connected-component size for RBPMS mask, in pixels |
| `size2` | `20000` | Minimum connected-component size for tissue mask, in pixels |
| Gaussian smoothing `sigma` | `5` | Smoothing for both masks, in pixels |
| Morphological opening | `7 × 7` | Structuring-element shape |
| `gridsize` | `70` | Hexbin spatial resolution |
| `color` | `magma` | Ratio-map colormap |
| `bg_ratio` | `0.07` | Intensity assigned to tissue-context layer |
| Ratio display range | `0–1` | Fixed colormap limits (`set_clim(0, 1)`) |

Parameters used for segmentation may require adjustment for new images. For comparisons between samples, keep binning and display ranges consistent and inspect the intermediate masks.

## Output and interpretation

| Output | Description |
| --- | --- |
| `output_hexbin_plot/Hexbin_map.pdf` | Spatial hexbin ratio visualization in PDF format |
| `output_hexbin_plot/Hexbin_map.svg` | Spatial hexbin ratio visualization in SVG format |

The notebook also displays intermediate image and mask previews and prints background estimates. It does not export per-cell measurements, per-bin numerical ratio tables, or statistical summaries.

The result is a ratio of *mean fluorescence values within hexagonal bins of RBPMS-positive pixels*; it is not necessarily an individual-cell measurement. The RBPMS-positive mask and background estimates should be checked against the experimental images before interpreting differences between samples.

## Notes

- Prepare and align the paired images before running the notebook. Its documentation describes cropping and excluding non-retinal regions using Fiji/ImageJ's *Clear Outside* workflow.
- Confirm that image dimensions match before segmentation and that the tissue mask is non-empty for median background estimation.
- Do not interpret this pipeline as the separate calculation of the percentage of p-S6-positive RGCs.
- The supplied TIFFs are inputs to this analysis; the folder does not establish that they are the original microscope acquisition files.
