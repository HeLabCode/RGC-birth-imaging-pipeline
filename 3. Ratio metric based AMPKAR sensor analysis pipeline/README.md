# Per-View Ratiometric Analysis Pipeline (ExRai-AMPKAR)

## Overview

This pipeline performs single-cell segmentation, Z-stack tracking, cross-excitation cell matching, quality control, and ratiometric signal extraction from two-photon images acquired at **800 nm and 920 nm excitation** through a common emission channel.

The central notebook processes one imaging **view** at a time. It uses an included pretrained segmentation model, interactive/manual checks, and step-specific Python utilities. The output includes background-corrected intensity measurements, an **920/800 excitation ratio** per matched cell, and pseudocolor visualization files. The example demonstrates the analysis workflow rather than a complete multi-sample statistical analysis.

## Repository organization

| Location | Purpose |
| --- | --- |
| [`ExRai AMPKAR analysis/per_view_analysis.ipynb`](ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb) | Main notebook; run steps S0–S7 for a single view |
| [`ExRai AMPKAR analysis/NN_model/`](ExRai%20AMPKAR%20analysis/NN_model/) | Included `NN_pretrained_model.pth` used for initial segmentation |
| [`ExRai AMPKAR analysis/S1_NN_segmentation/`](ExRai%20AMPKAR%20analysis/S1_NN_segmentation/) | Neural-network segmentation utilities |
| [`ExRai AMPKAR analysis/S1_regiongrow/`](ExRai%20AMPKAR%20analysis/S1_regiongrow/) | Interactive cell refinement / region growing |
| [`ExRai AMPKAR analysis/S2_tracking/`](ExRai%20AMPKAR%20analysis/S2_tracking/) | Z-stack alignment and cell tracking |
| [`ExRai AMPKAR analysis/S3_mapping/`](ExRai%20AMPKAR%20analysis/S3_mapping/) | 800 nm / 920 nm cell correspondence |
| [`ExRai AMPKAR analysis/S4_illumination_drop_out/`](ExRai%20AMPKAR%20analysis/S4_illumination_drop_out/) | Illumination-based filtering |
| [`ExRai AMPKAR analysis/S5_background_subtraction/`](ExRai%20AMPKAR%20analysis/S5_background_subtraction/) | Per-cell local background correction |
| [`ExRai AMPKAR analysis/S6_size_ratio_subtraction/`](ExRai%20AMPKAR%20analysis/S6_size_ratio_subtraction/) | Mask-area consistency filtering |
| [`ExRai AMPKAR analysis/S7_final_output/`](ExRai%20AMPKAR%20analysis/S7_final_output/) | Final per-cell ratio and projection generation |
| [`ExRai AMPKAR analysis/common_functions/`](ExRai%20AMPKAR%20analysis/common_functions/) | Shared loading, geometry, and display utilities |
| [`Example images and output/view1/`](Example%20images%20and%20output/view1/) | Example excitation-paired TIFF images and saved intermediate/final outputs |

## Example dataset and experimental specificity

The supplied `view1` example consists of **11 input TIFF slices** (not counting derived TIFFs in `results/`):

| Image series | Files | Image properties |
| --- | --- | --- |
| 800 nm excitation | `800_ch2_Z001.tif` through `800_ch2_Z006.tif` (6 slices) | 512 × 512 px, single-channel, 8-bit |
| 920 nm excitation | `920_ch2_Z001.tif` through `920_ch2_Z005.tif` (5 slices) | 512 × 512 px, single-channel, 8-bit |

The unequal numbers of slices are present in the supplied example and are discovered independently by the notebook. The code's input-file convention is `800_ch2_Zxxx.tif` / `920_ch2_Zxxx.tif`, **not** `800_ch02_T***.tif` / `920_ch02_T***.tif` from the earlier README.

The paired TIFFs represent the two excitation series used by the analysis notebook. They are input files to this workflow; the archive alone does not establish whether they are original instrument-native recordings or processed exports.

## Requirements

Install Python/Jupyter and the notebook's dependencies:

```bash
pip install numpy scipy scikit-image opencv-python tifffile matplotlib torch segmentation-models-pytorch ipywidgets trackpy
```

Run the notebook in an environment with functioning interactive Jupyter widgets. A compatible PyTorch checkpoint is included in `ExRai AMPKAR analysis/NN_model/`. The archive does not include pinned package versions.

## Running the example analysis

- [ ] Download the repository and keep the `ExRai AMPKAR analysis/` and `Example images and output/` directories together.
- [ ] Open [`per_view_analysis.ipynb`](ExRai%20AMPKAR%20analysis/per_view_analysis.ipynb) in Jupyter, with **`ExRai AMPKAR analysis/` as the notebook's working directory** so local Python imports and the model path resolve.
- [ ] In the notebook's **Load datasets** cell, set `VIEW_DIR` to the local path of `Example images and output/view1/`. Its current default is `Path.cwd().parent / "Example images and output/view1"`.
- [ ] Run S0 and check that both excitation series are found; inspect the Z-stacks for image quality and correspondence.
- [ ] Run S1.1 to generate initial neural-network segmentations for both excitations, then perform S1.2 manual region-growing segmentation/refinement as needed.
- [ ] Run S2 for both excitations, selecting the **three manual anchor points per slice** required by the tracking interface.
- [ ] Run S3 and confirm cell matches between the 800 nm and 920 nm stacks; review mapping parameters and manual interactions.
- [ ] Run S4 illumination filtering, S5 local background subtraction, and S6 area-consistency filtering, checking the intermediate outputs.
- [ ] Run S7 and inspect the CSV results and pseudocolor projections in `view1/results/S7_final_output/`.

**Important:** This is an interactive, multi-stage pipeline, not an unattended batch process. The archive includes previously generated outputs in `view1/results/`; rerunning cells may overwrite or change those outputs. Copy the example view to a separate working folder before rerunning if you want to preserve the supplied reference files.

## Analysis workflow and outputs

| Stage | Operation | Example saved result |
| --- | --- | --- |
| S0 | Discover and inspect both TIFF stacks | `results/S0_discovery/discovery.json` |
| S1.1 | Pretrained U-Net/NN segmentation | `results/S1_NN_seg_800/`, `results/S1_NN_seg_920/` |
| S1.2 | Interactive region growing and mask refinement | `results/S1_seg_800/`, `results/S1_seg_920/` |
| S2 | Align and track cells through Z | `results/S2_tracks_800/`, `results/S2_tracks_920/` |
| S3 | Map cells between excitations | `results/S3_mapping_results/Matching_map.csv` |
| S4 | Filter by excitation illumination | `results/S4_illumination_correction/illumination_mapping_filtered_dual.csv` |
| S5 | Extract raw and local-background-subtracted mean intensity | `results/S5_backg_subtraction_800/800_S5_means_bgsub.csv`, `results/S5_backg_subtraction_920/920_S5_means_bgsub.csv` |
| S6 | Remove size-inconsistent mapped cells | `results/S6_size_ratio_dropout/area_filtered_mapping.csv` |
| S7 | Compute 920/800 ratio and generate visualizations | `results/S7_final_output/S7_manual_ratio_results.csv`, `S7_ratio_projection.tif`, `S7_ratio_projection_nolabels.tif`, `S7_colorbar.tif` |

S7 also uses the filename `view1.pkl` for a saved view-specific file. This fixed name can overwrite existing output when reused.

## Key parameters and validation

- **S1:** segmentation/model thresholds and the manual region-growing decisions.
- **S2:** manually selected Z-slice anchor points and tracking decisions.
- **S3:** matching method (`Hungarian` or `Nearest`), maximum centroid distance, and X/Y offsets. The example widget initializes distance at **15 px**, but this is adjustable.
- **S4:** illuminated-region thresholds / exclusion criteria.
- **S5:** local background estimation and subtraction settings.
- **S6:** cross-channel segmentation area consistency.
- **S7:** ratio display bounds and output naming.

Check masks, tracking, matching, and background quality for every new dataset. The notebook computes **920 nm / 800 nm** from background-corrected intensities and handles invalid denominators. Comparisons require a consistent ratio definition.

## Notes

- For new datasets, preprocessing must produce correctly named excitation-paired TIFF slices whose Z indices start at `Z001`, with no skipped initial slices.
- The example contains 6 slices at 800 nm and 5 at 920 nm; check cross-stack correspondence explicitly rather than assuming identical depth.
- The segmentation checkpoint is approximately **130 MB**, exceeding GitHub's 100 MiB limit for ordinary Git files.
- The analysis includes manual decisions; the included outputs are saved reference files, not evidence of a fresh independent rerun.
