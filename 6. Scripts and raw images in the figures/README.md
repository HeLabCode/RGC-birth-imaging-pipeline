# Example Analysis Visualizations

This folder contains two notebooks for visualizing prepared fluorescence-imaging analysis results. Each notebook has its own README, input dataset, and example output files.

| Workflow | Notebook | Input | Output |
| --- | --- | --- | --- |
| ExRai-AMPKAR Z-stack video and intensity heatmaps | [ExRai notebook](Figure3b%20-%20ExRai-AMPKAR%20analysis%20video%20plot%20/ExRai_AMPKAR_heatmaps_video.ipynb) | TIFF Z-planes and saved S2/S5/S7 results | AVI videos and peak-alignment CSV tables |
| pHLemon-rev FRET video | [FRET notebook](Figure5b%20-%20pHLr-FRET%20sensor%20video%20plot/pHLr_FRET_video.ipynb) | Donor/acceptor TIFFs, masks, and precomputed FRET table | MP4 video and mean-FRET CSV |

## Usage

Open the relevant subfolder and follow its README:

- [ExRai-AMPKAR visualization README](Figure3b%20-%20ExRai-AMPKAR%20analysis%20video%20plot%20/README.md)
- [pHLemon-rev FRET visualization README](Figure5b%20-%20pHLr-FRET%20sensor%20video%20plot/README.md)

Run each notebook from its own folder, since its default input and output paths are relative to the current working directory. Both notebooks visualize supplied analysis results; they do not re-run the upstream segmentation or fluorescence-quantification pipelines.
