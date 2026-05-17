# RGC-birth-imaging-pipeline

## Overview
This repository contains a complete set of analysis pipelines for retinal imaging experiments, focused on cell segmentation, tracking, and ratiometric/FRET-based signal analysis across multiple experimental contexts.

The codebase consolidates independent workflows into a single structured repository, covering both in vitro and in vivo imaging, static and time-lapse data.

---

## Repository Structure

The repository is organized into modular folders, each corresponding to a specific analysis workflow:

- **ExRai-AMPKAR analysis pipeline**  
  Ratiometric analysis pipeline for AMPKAR-based imaging experiments.

- **FRET-based sensor analysis pipeline**  
  Per-cell FRET efficiency analysis from paired donor and acceptor fluorescence images using a binary segmentation mask. Includes signal extraction, EFRET calculation, pseudocolor visualization, and CSV output.

- **Hexbin plot for p-S6 signal in retina section**  
  Spatial aggregation and visualization of pS6 signal using hexbin-based ratio maps with preserved tissue context.

- **In vitro analysis with Cellpose**  
  Segmentation and analysis of cultured cells using Cellpose-based workflows.

- **U-net based cell segmentation**  
  Full training and optimization pipeline for deep learning–based segmentation models, including hyperparameter search and evaluation.

---

## Scope

Across all modules, the repository covers:

- Image preprocessing and normalization  
- Cell segmentation
- Manual and semi-automated refinement  
- Spatial analysis and visualization  
- Time-lapse drift correction and tracking  
- Ratiometric and FRET-based signal extraction 

---

## Design Principles

- **Modular**: each pipeline is self-contained and reusable  
- **Reproducible**: consistent parameter control and documented workflows  
- **Interactive where needed**: manual steps included for quality control  
- **Scalable**: supports both small datasets and larger experiments  

---

## Contact

For questions, please contact Zhigang He at Zhigang.He@childrens.harvard.edu
