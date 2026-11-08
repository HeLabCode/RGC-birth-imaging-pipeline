# RGC Birth Imaging and Analysis Pipelines

**Analysis code for:** *The Onset of Breathing at Birth Switches Off Regenerative Capacity*  
Xuan Song, Xinyi Cai, Shane V. Hegarty, Davide Speri, *et al.*

## Overview

This repository contains image-processing, quantitative-analysis, and visualization workflows developed for experiments investigating how the onset of breathing regulates regenerative signaling in retinal ganglion cells (RGCs).

The accompanying study identifies a CO₂/pH–SNRK–mTOR signaling pathway linking postnatal respiration to the rapid decline in neuronal growth-associated signaling. The code supports measurements from *in vivo* two-photon imaging, genetically encoded fluorescent reporters, retinal immunofluorescence, and cell segmentation.

**Five analysis workflows and two figure-specific video workflows** are provided. The analysis folders and both video subfolders have their own READMEs describing inputs, dependencies, workflows, and outputs.

## Repository organization

| Module | Description | Main notebook |
| --- | --- | --- |
| **ExRai-AMPKAR analysis pipeline** | Single-cell ratiometric analysis of paired 800/920 nm recordings, including segmentation, tracking, matching, background subtraction, and quality control | [ExRai-AMPKAR analysis notebook](3.%20Ratio%20metric%20based%20AMPKAR%20sensor%20analysis%20pipeline/ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb) |
| **Figure 3b – ExRai-AMPKAR video plot** | Z-stack videos combining excitation images, per-cell ratio projections, intensity heatmaps, and peak-plane highlights | [ExRai-AMPKAR video notebook](6.%20Scripts%20and%20raw%20images%20in%20the%20figures/Figure3b%20-%20ExRai-AMPKAR%20analysis%20video%20plot%20/ExRai_AMPKAR_heatmaps_video.ipynb) |
| **FRET-based sensor analysis pipeline** | Extraction of donor/acceptor fluorescence signals and per-cell FRET efficiency calculation | [FRET analysis notebook](4.%20FRET%20based%20sensor%20analysis%20pipeline/FRET_analysis.ipynb) |
| **Figure 5b – pHLr-FRET sensor video plot** | Video visualization of prepared donor/acceptor images, cell FRET maps, and mean FRET time courses | [pHLr FRET video notebook](6.%20Scripts%20and%20raw%20images%20in%20the%20figures/Figure5b%20-%20pHLr-FRET%20sensor%20video%20plot/pHLr_FRET_video.ipynb) |
| **p-S6 hexbin analysis** | Spatial visualization of p-S6/RBPMS fluorescence relationships in retinal sections | [p-S6 hexbin notebook](1.%20Hexbin%20plot%20for%20p-S6%20signal%20in%20retina%20section/Hexbin_plot_code.ipynb) |
| **Cellpose analysis** | Cell segmentation and fluorescence-ratio quantification from cultured-cell maximum-intensity projections | [Cellpose notebook](5.%20Cellpose%20based%20in%20vitro%20imaging%20analysis%20pipeline/Cellpose_segmentation%26analysis.ipynb) |
| **U-Net segmentation** | Training, evaluation, and application of deep-learning models for cell segmentation | [U-Net training notebook](2.%20U-net%20based%20cell%20segmentation/2.2%20Selection%20for%20best%20NN%20parameters/U_net_train_model_hparam_search.ipynb.ipynb); [U-Net inference notebook](2.%20U-net%20based%20cell%20segmentation/2.3%20Running%20Neural%20Network/NN_binary_mask.ipynb) |

## Getting started

### 1. Download the code

Clone or download the repository and open the folder for the analysis or visualization you wish to run.

### 2. Set up the environment

The workflows use Python and Jupyter notebooks, with dependencies specific to each module. Follow the **Requirements** section of the corresponding README.

```bash
jupyter lab
```

### 3. Run an analysis or visualization

1. Open the relevant notebook.
2. Locate its input and output directory settings.
3. Use the provided input data or point the notebook to a compatible dataset.
4. Execute the cells in order and review the resulting figures, tables, and quality-control outputs.

The figure-video notebooks use **previously generated analysis results** and do not repeat the upstream segmentation or quantitative analysis. Some upstream workflows require manual segmentation refinement, image registration, or cell matching.

## Code associated with manuscript figures

The table identifies the analysis or visualization workflows associated with the listed manuscript figures. These links identify relevant code; they do not establish that the supplied examples reproduce each complete panel. Full panel reproduction may require additional experimental data and downstream analysis.

| Manuscript figure | Data or analysis | Relevant code |
| --- | --- | --- |
| **Fig. 1a–c** | Spatial p-S6/RBPMS maps in retinal sections | [Hexbin notebook](1.%20Hexbin%20plot%20for%20p-S6%20signal%20in%20retina%20section/Hexbin_plot_code.ipynb) |
| **Fig. 3b** | ExRai-AMPKAR Z-stack and ratio visualization | [ExRai analysis](3.%20Ratio%20metric%20based%20AMPKAR%20sensor%20analysis%20pipeline/ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb); [Figure 3b video](6.%20Scripts%20and%20raw%20images%20in%20the%20figures/Figure3b%20-%20ExRai-AMPKAR%20analysis%20video%20plot%20/ExRai_AMPKAR_heatmaps_video.ipynb) |
| **Fig. 3c–e** | ExRai-AMPKAR kinase-activity analysis, including *Snrk* loss of function | [ExRai analysis](3.%20Ratio%20metric%20based%20AMPKAR%20sensor%20analysis%20pipeline/ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb) |
| **Fig. 4b, d** | ExRai-AMPKAR responses under altered CO₂ or O₂ | [ExRai analysis](3.%20Ratio%20metric%20based%20AMPKAR%20sensor%20analysis%20pipeline/ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb) |
| **Fig. 5b** | pHLemon-rev FRET sensor visualization around breathing onset | [FRET analysis](4.%20FRET%20based%20sensor%20analysis%20pipeline/FRET_analysis.ipynb); [Figure 5b video](6.%20Scripts%20and%20raw%20images%20in%20the%20figures/Figure5b%20-%20pHLr-FRET%20sensor%20video%20plot/pHLr_FRET_video.ipynb) |
| **Fig. 5a, c–d** | pHLemon-rev intracellular-pH measurements | [FRET analysis](4.%20FRET%20based%20sensor%20analysis%20pipeline/FRET_analysis.ipynb) |
| **Fig. 6b–c** | Adult RGC intracellular-pH and AMPK-family reporter measurements | [FRET analysis](4.%20FRET%20based%20sensor%20analysis%20pipeline/FRET_analysis.ipynb); [ExRai analysis](3.%20Ratio%20metric%20based%20AMPKAR%20sensor%20analysis%20pipeline/ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb) |
| **Extended Data Fig. 13** | ATP and calcium reporter measurements | [FRET analysis](4.%20FRET%20based%20sensor%20analysis%20pipeline/FRET_analysis.ipynb) |
| **Extended Data Fig. 14** | Deep-learning segmentation and comparison with annotated cells | [U-Net training](2.%20U-net%20based%20cell%20segmentation/2.2%20Selection%20for%20best%20NN%20parameters/U_net_train_model_hparam_search.ipynb.ipynb); [U-Net inference](2.%20U-net%20based%20cell%20segmentation/2.3%20Running%20Neural%20Network/NN_binary_mask.ipynb) |
| **Extended Data Figs. 15–18** | FRET correction, reporter calibration, and sensor-validation analyses | [Cellpose notebook](5.%20Cellpose%20based%20in%20vitro%20imaging%20analysis%20pipeline/Cellpose_segmentation%26analysis.ipynb) |
## Analysis modules

### ExRai-AMPKAR: ratiometric two-photon imaging

**Input:** Paired 800 nm and 920 nm TIFF image stacks from the same field of view.  
**Analysis:** Segmentation → Z-tracking → channel matching → illumination filtering → background subtraction → per-cell excitation ratio.  
**Output:** Per-cell measurements, pseudocolor projections, and quality-control files within dataset-specific `results/` directories.

**Notebook:** [ExRai-AMPKAR analysis notebook](3.%20Ratio%20metric%20based%20AMPKAR%20sensor%20analysis%20pipeline/ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb)

### Figure 3b: ExRai-AMPKAR videos and intensity heatmaps

**Input:** `Input_analysis_results/` containing the `0min/`, `1min/`, `3min/`, `6min/`, and `8min/` folders, with paired TIFF stacks and previously generated S2 masks, S5 intensity tables, and S7 ratios.  
**Visualization:** Three-panel Z-stack videos, matched-cell intensity heatmaps, and channel-specific peak-plane highlights.  
**Output:** AVI videos and peak-alignment CSV files in `output_folder/`.

**Notebook:** [ExRai-AMPKAR video notebook](6.%20Scripts%20and%20raw%20images%20in%20the%20figures/Figure3b%20-%20ExRai-AMPKAR%20analysis%20video%20plot%20/ExRai_AMPKAR_heatmaps_video.ipynb)

### FRET-based biosensors

**Input:** Aligned donor and acceptor fluorescence images and a binary cell mask.  
**Analysis:** Segmentation-based fluorescence extraction → per-cell FRET efficiency calculation → pseudocolor visualization.  
**Output:** `cell_FRET_efficiency.csv`, `EFRET_per_cell_pseudocolor.tif`, and segmentation previews.

**Notebook:** [FRET analysis notebook](4.%20FRET%20based%20sensor%20analysis%20pipeline/FRET_analysis.ipynb)

The manuscript describes additional bleed-through correction and photon-flux normalization; consult the Methods and corresponding code when interpreting calibrated measurements.

### Figure 5b: pHLemon-rev FRET sensor video

**Input:** `Input_analysis_results/` containing representative cropped donor and acceptor images, previously generated neural-network cell masks, and `cell_FRET.csv`.  
**Visualization:** Per-cell FRET pseudocolor images combined with donor/acceptor fluorescence and the mean FRET time course.  
**Output:** `pHLr_analysis_video_results.mp4` and `mean_FRET_per_view.csv` in `output_exported_video/`.

**Notebook:** [pHLr FRET video notebook](6.%20Scripts%20and%20raw%20images%20in%20the%20figures/Figure5b%20-%20pHLr-FRET%20sensor%20video%20plot/pHLr_FRET_video.ipynb)

The images are representative crops of a larger field of view. Segmentation was performed on the original full-size images, and FRET values were calculated before video generation. The supplied frames are not necessarily consecutive: frames affected by breathing-related displacement or unstable movement were excluded. Interpolation, where used for visualization, does not recover excluded measurements.

**Upstream notebooks:** [U-Net inference notebook](2.%20U-net%20based%20cell%20segmentation/2.3%20Running%20Neural%20Network/NN_binary_mask.ipynb) and [FRET analysis notebook](4.%20FRET%20based%20sensor%20analysis%20pipeline/FRET_analysis.ipynb).

### p-S6/RBPMS hexbin visualization

**Input:** Aligned, cropped retinal p-S6 and RBPMS TIFF images.  
**Analysis:** RBPMS segmentation → tissue masking → background subtraction → hexagonal aggregation → p-S6/RBPMS ratio visualization.  
**Output:** Spatial maps in PDF and SVG format.

**Notebook:** [p-S6 hexbin notebook](1.%20Hexbin%20plot%20for%20p-S6%20signal%20in%20retina%20section/Hexbin_plot_code.ipynb)

### Cellpose-based single-cell analysis

**Input:** Paired, aligned maximum-intensity-projection TIFF images.  
**Analysis:** Cellpose segmentation → per-cell channel intensities → fluorescence ratios → quality filtering.  
**Output:** Segmentation masks, overlays, CSV tables, and intensity-distribution plots.

**Notebook:** [Cellpose notebook](5.%20Cellpose%20based%20in%20vitro%20imaging%20analysis%20pipeline/Cellpose_segmentation%26analysis.ipynb)

### U-Net cell segmentation

**Input:** Microscopy images, matching manual masks, and image-group metadata.  
**Analysis:** Group-aware splitting → augmentation → U-Net training and hyperparameter search → evaluation → mask prediction.  
**Output:** Model checkpoints, evaluation metrics, and predicted masks.

**Notebooks:** [U-Net training notebook](2.%20U-net%20based%20cell%20segmentation/2.2%20Selection%20for%20best%20NN%20parameters/U_net_train_model_hparam_search.ipynb.ipynb) and [U-Net inference notebook](2.%20U-net%20based%20cell%20segmentation/2.3%20Running%20Neural%20Network/NN_binary_mask.ipynb)

## Notes on reproducibility

- Each module's README specifies its input format, directory structure, dependencies, and expected outputs.
- Figure-specific video notebooks use supplied, previously processed analysis results; they are not replacements for the underlying segmentation and quantification workflows.
- Segmentation, image alignment, background correction, cell matching, and quality filtering should be verified when applying the pipelines to new datasets.
- Keep display ranges consistent when comparing pseudocolor images or videos across time points.

## Citation

If you use this code, please cite the associated manuscript:

Song, X., Cai, X., Hegarty, S. V., Speri, D., *et al.* **The Onset of Breathing at Birth Switches Off Regenerative Capacity.**

## Contact

Questions regarding code: **Zhigang He** (Zhigang.He@childrens.harvard.edu).
