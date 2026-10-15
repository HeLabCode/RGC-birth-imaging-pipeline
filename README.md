# RGC Birth Imaging and Analysis Pipelines

**Analysis code for:** *The Onset of Breathing at Birth Switches Off Regenerative Capacity*  
Xuan Song, Xinyi Cai, Shane V. Hegarty, Davide Speri, *et al.*

## Overview

This repository contains image-processing and quantitative-analysis workflows developed for experiments investigating how the onset of breathing regulates regenerative signaling in retinal ganglion cells (RGCs).

The accompanying study identifies a CO₂/pH–SNRK–mTOR signaling pathway linking postnatal respiration to the rapid decline in neuronal growth-associated signaling. The analysis code supports measurements from *in vivo* two-photon imaging, genetically encoded fluorescent reporters, retinal immunofluorescence, and cell segmentation.

Five analysis workflows are provided. Each has its own README with input formats, dependencies, and step-by-step instructions.

## Repository organization

| Analysis module | Description | Main notebook |
| --- | --- | --- |
| **ExRai-AMPKAR analysis** | Single-cell ratiometric analysis of paired 800/920 nm two-photon recordings, including segmentation, tracking, channel matching, background correction, and quality control | `ExRai_ratio_video_analysis/per_view_analysis.ipynb` |
| **FRET sensor analysis** | Extraction of donor/acceptor fluorescence signals and calculation of per-cell FRET efficiency | `FRET_analysis.ipynb` |
| **p-S6 hexbin analysis** | Spatial visualization of p-S6/RBPMS fluorescence relationships in retinal sections | `Hexbin_plot_code.ipynb` |
| **Cellpose analysis** | Cell segmentation and fluorescence-ratio quantification from cultured-cell maximum-intensity projections | `Cellpose_segmentation&analysis.ipynb` |
| **U-Net segmentation** | Training, testing, and application of deep-learning models for cell segmentation | `U_net_train_model_hparam_search.ipynb`, `Running_NN/NN_binary_mask.ipynb` |


## Getting started

### 1. Download the code

Clone or download this repository and open the folder for the analysis you wish to perform.

### 2. Set up the environment

The pipelines use Python and Jupyter notebooks, with additional dependencies specific to each analysis. Follow the **Requirements** section of the relevant module README rather than assuming that all notebooks use the same environment.

```bash
jupyter lab
```

### 3. Run an example analysis

1. Open the module's main notebook.
2. Locate the configuration or file-path section.
3. Point the notebook to the corresponding example images, where included.
4. Execute the cells in order and review intermediate segmentation and quality-control outputs.
5. Inspect the saved results in the output directory described by that module's README.

Some workflows require interactive segmentation refinement, image registration, or manual validation.

## Code associated with manuscript figures

The table below connects the **analysis underlying specific figure panels** to the relevant code. The notebooks process images and extract quantitative measurements; generation of an entire manuscript panel may additionally require the original experimental dataset, normalization, statistical analysis, and figure assembly.

| Manuscript figure | Data or analysis | Relevant code |
| --- | --- | --- | 
| **Fig. 1a–c** | Maps of p-S6 fluorescence relative to RBPMS signal in retinal sections | Hexbin plot for p-S6 signal in retina section |
| **Fig. 3b–e** | ExRai-AMPKAR imaging of kinase activity at birth, including *Snrk* loss of function | ExRai-AMPKAR analysis pipeline|
| **Fig. 4b, d** | ExRai-AMPKAR responses under altered CO₂ or O₂ | ExRai-AMPKAR analysis pipeline| 
| **Fig. 5a–d** | pHLemon-rev measurements of intracellular pH around breathing onset | FRET-based sensor analysis pipeline| 
| **Fig. 6b–c** | Adult RGC intracellular-pH and AMPK-family reporter measurements | FRET-based sensor analysis pipeline & ExRai-AMPKAR analysis pipeline  |
| **Extended Data Fig. 13** | ATP and calcium reporter measurements | FRET-based sensor analysis pipeline |
| **Extended Data Fig. 14** | Deep-learning segmentation and comparison with annotated cells | U-Net based cell segmentation | 
| **Extended Data Figs. 15–18** | FRET correction, reporter calibration, and sensor-validation analyses | In vitro imaging analysis with Cellpose|

For each figure, use the notebook named above to identify the input format and analysis steps, then consult its dedicated README. The table identifies the corresponding *analysis workflow*.

## Analysis modules

### ExRai-AMPKAR: ratiometric two-photon imaging

**Input:** Paired TIFF stacks from the same field of view (`800_ch02_T***.tif` and `920_ch02_T***.tif`).  
**Analysis:** Segmentation → cell tracking → channel matching → illumination filtering → background subtraction → single-cell ratio calculation.  
**Output:** Per-cell measurements, pseudocolor visualizations, and quality-control files in a dataset-specific `results/` folder.

**Notebook:** `ExRai_ratio_video_analysis/per_view_analysis.ipynb`

### FRET-based biosensors

**Input:** Aligned donor and acceptor images and a binary cell mask (`donor.tif`, `acceptor.tif`, `binary_mask.tif`).  
**Analysis:** Mask-based fluorescence extraction → FRET-efficiency calculation → visualization.  
**Output:** `cell_FRET_efficiency.csv`, `EFRET_per_cell_pseudocolor.tif`, and segmentation previews.

**Notebook:** `FRET_analysis.ipynb`

The manuscript describes additional bleed-through correction and photon-flux normalization; the relevant Methods and code should be consulted when interpreting calibrated measurements.

### p-S6/RBPMS hexbin visualization

**Input:** Aligned, cropped retinal p-S6 and RBPMS TIFF images.  
**Analysis:** RBPMS segmentation → tissue masking → background subtraction → hexagonal aggregation → p-S6/RBPMS ratio visualization.  
**Output:** Spatial maps in PDF/SVG format.

**Notebook:** `Hexbin_plot_code.ipynb`

### Cellpose-based single-cell analysis

**Input:** Paired, aligned maximum-intensity-projection TIFFs (`ch00_MIP.tif`, `ch01_MIP.tif`).  
**Analysis:** Cellpose segmentation → per-cell channel intensities → fluorescence ratios → quality filtering.  
**Output:** Segmentation masks, overlay images, CSV tables, and ratio-distribution plots.

**Notebook:** `Cellpose_segmentation&analysis.ipynb`

### U-Net cell segmentation

**Input:** Microscopy images, matched manual masks, and image-group metadata.  
**Analysis:** Group-aware splitting → data augmentation → U-Net training and hyperparameter optimization → held-out evaluation → segmentation prediction.  
**Output:** Model checkpoints (including `best_semantic_model.pth`), evaluation metrics, and predicted masks.

**Notebooks:** `U_net_train_model_hparam_search.ipynb`; `Running_NN/NN_binary_mask.ipynb`

## Notes on reproducibility

- Each module's README specifies its expected filenames, directory structure, and dependencies.
- Segmentation, alignment, background correction, and filtering should be checked for each new dataset.

## Citation

If you use this code, please cite the associated manuscript:

Song, X., Cai, X., Hegarty, S. V., Speri, D., *et al.* **The Onset of Breathing at Birth Switches Off Regenerative Capacity.** 

## Contact

Questions regarding code or requests for figure-associated datasets: **Zhigang He** (Zhigang.He@childrens.harvard.edu).
