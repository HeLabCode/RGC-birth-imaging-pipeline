# Example Analysis Visualizations

This folder contains two notebooks for visualizing prepared fluorescence-imaging analysis results. Each notebook has its own README, input dataset, and example output files.

| Workflow | Notebook | Input | Output |
| --- | --- | --- | --- |
| ExRai-AMPKAR Z-stack video and intensity heatmaps | [ExRai notebook](Figure3b%20-%20ExRai-AMPKAR%20analysis%20video%20plot%20code%20/ExRai_AMPKAR_heatmaps_video.ipynb) | TIFF Z-planes and saved S2/S5/S7 results | AVI videos and peak-alignment CSV tables |
| pHLemon-rev FRET video | [FRET notebook](Figure5d%20-%20pHLr-FRET%20sensor%20video%20plot%20code/pHLr_FRET_video.ipynb) | Donor/acceptor TIFFs, masks, and precomputed FRET table | MP4 video and mean-FRET CSV |

## Usage

Open the relevant subfolder and follow its README:

- [ExRai-AMPKAR visualization README](Figure3b%20-%20ExRai-AMPKAR%20analysis%20video%20plot%20code%20/README.md)
- [pHLemon-rev FRET visualization README](Figure5d%20-%20pHLr-FRET%20sensor%20video%20plot%20code/README.md)

Run each notebook from its own folder, since its default input and output paths are relative to the current working directory. Both notebooks visualize supplied analysis results; they do not re-run the upstream segmentation or fluorescence-quantification pipelines.

## Supplementary videos

[Supplementary Videos](Supplementary%20Videos/README.md) contains the final MP4s for Fig. 3b (wild type), Fig. 3d (*Snrk* knockout), Fig. 4a (10% CO₂), Fig. 4c (10% O₂), and Fig. 5d (pHLemon-rev FRET). The README includes the complete legends and maps the figure-based filenames to Supplementary Videos 3–7.
