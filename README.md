# RGC-birth-imaging-pipeline

## Overview
This repository contains a complete set of analysis pipelines for retinal imaging experiments, focused on cell segmentation, tracking, and ratiometric signal analysis across multiple experimental contexts.

The codebase consolidates independent workflows into a single structured repository, covering both in vitro and in vivo imaging, static and time-lapse data, and classical as well as deep learning–based approaches.

---

## Repository Structure

The repository is organized into modular folders, each corresponding to a specific analysis workflow:

- **ExRai-AMPKAR analysis pipeline**  
  Ratiometric analysis pipeline for AMPKAR-based imaging experiments.

- **FRET-based imaging analysis**  
  Per-view pipeline for extracting excitation ratios from two-photon datasets (800 nm / 920 nm), including segmentation, tracking, and filtering.

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
- Cell segmentation (classical + deep learning)  
- Manual and semi-automated refinement  
- Spatial analysis and visualization  
- Time-lapse drift correction and tracking  
- Ratiometric signal extraction  
- Model training and evaluation  

---

## Design Principles

- **Modular**: each pipeline is self-contained and reusable  
- **Reproducible**: consistent parameter control and documented workflows  
- **Interactive where needed**: manual steps included for quality control  
- **Scalable**: supports both small datasets and larger experiments  

---

## Requirements

Each subfolder contains its own dependencies.  
Typical required libraries include:
