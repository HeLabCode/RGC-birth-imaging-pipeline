# U-Net Cell Segmentation Training Pipeline

## Overview
This notebook trains, optimizes, and evaluates a supervised U-Net model for cell segmentation in two-photon microscopy images.

The workflow includes dataset loading, group-aware splitting, preprocessing, augmentation, staged transfer learning, hyperparameter optimization, threshold selection, held-out test evaluation, and qualitative inspection of segmentation outputs.

The pipeline is designed to train a segmentation model that generalizes across heterogeneous imaging conditions while reducing information leakage between related fields of view.

---

## Features
- Group-aware train/validation/test splitting
- Image and mask preprocessing
- Data augmentation with Albumentations
- U-Net segmentation model with ResNet-50 encoder
- ImageNet-pretrained encoder weights
- Composite BCE + Dice loss
- Two-stage transfer learning:
  - frozen encoder training
  - full-model fine-tuning
- Early stopping based on validation Dice
- Optuna hyperparameter search
- Validation-based threshold selection
- Held-out test evaluation
- Per-image metric reporting
- Qualitative prediction visualization
- Optional data-efficiency analysis

---

## Repository Structure

### `example_images_full_dataset/`
Raw input images (100 images).
- Heterogeneous dataset used for training and evaluation

### `manual_masks_full_dataset/`
Manually annotated ground-truth masks.
- Binary TIFF masks created by hand

### `image_similarity_groups.csv`
Metadata file defining groups of similar images.
- Used for group-aware train/validation/test splitting

### `NN&manual_comparison/`
Qualitative comparison between manual annotations and model predictions.
- Includes zoomed examples used for visualization (figures in the paper)

### `semantic_segmentation_output/`
Outputs from the trained U-Net model.

- Contains the best saved model (`best_semantic_model.pth`)
- Includes final evaluation files, plots, and training summaries
- Also contains `hyperparameter_search/`, where the optimization runs are stored

### `semantic_segmentation_output/hyperparameter_search/`
Hyperparameter search results.

- Stores training trial outputs
- The best run is included here
- Additional runs can be provided on request

### `U_net_train_model_hparam_search.ipynb`
Main notebook for training, optimization, and evaluation of the U-Net model

---



## Requirements

Install the required packages:

```bash
pip install numpy pandas matplotlib opencv-python torch scikit-learn albumentations segmentation-models-pytorch optuna natsort tifffile
```

---

## Input Data

The notebook expects paired image and mask datasets.

Typical inputs:
- Fluorescence microscopy images
- Binary segmentation masks
- Grouping table used for group-aware splitting

Images and masks should:
- Have matching filenames or a consistent naming scheme
- Be spatially aligned
- Represent the same field of view
- Be organized into directories defined in the configuration section

---

## Workflow

### 1. Configuration
Set:
- image directory
- mask directory
- metadata paths
- output directory
- random seed
- training constants

---

### 2. Reproducibility
The notebook seeds:
- Python
- NumPy
- PyTorch
- DataLoader workers

This helps make splitting, training, and optimization more reproducible.

---

### 3. Group-Aware Dataset Split
The dataset is split into:
- training set
- validation set
- held-out test set

Related fields of view are kept within the same split to reduce data leakage.

---

### 4. Preprocessing and Augmentation
Images are:
- converted to floating point
- resized
- normalized
- converted to tensors

Training data receives stochastic augmentation.  
Validation and test data use deterministic preprocessing.

---

### 5. Model Architecture
The model is a U-Net implemented with `segmentation-models-pytorch`.

Main settings:
- U-Net architecture
- ResNet-50 encoder
- ImageNet-pretrained encoder weights
- Single-channel binary segmentation output

---

### 6. Loss and Metrics
Training uses a combined loss:
- Binary cross-entropy with logits
- Dice loss

Metrics include:
- Dice
- IoU
- precision
- recall
- specificity
- accuracy

---

### 7. Two-Stage Training
Training is performed in two phases:

1. **Frozen encoder stage**
   - trains decoder and segmentation head

2. **Fine-tuning stage**
   - unfreezes the encoder
   - trains the full model

Validation Dice is used for model selection and early stopping.

---

### 8. Hyperparameter Optimization
Optuna is used to search over parameters such as:
- batch size
- learning rate
- freeze epochs
- fine-tuning epochs
- loss weights
- augmentation settings
- post-processing settings

Each trial trains a model and saves trial-level metrics.

---

### 9. Threshold Selection
The best model is evaluated across multiple binarization thresholds on the validation set.

The selected threshold is then used for final test evaluation.

---

### 10. Final Evaluation
The selected model is evaluated on the held-out test set.

Outputs include:
- final checkpoint
- training curves
- validation-threshold results
- test metrics
- per-image metrics
- example prediction visualizations

---

## Output

The pipeline saves:
- trained model checkpoints
- Optuna trial summaries
- best hyperparameters
- training history
- validation threshold table
- final test metrics
- per-image results
- diagnostic plots
- qualitative segmentation examples

---

## Optional: Data-Efficiency Analysis
The notebook can retrain the optimized pipeline on progressively larger subsets of the training data.

This helps estimate:
- how much annotation is needed
- whether performance has saturated
- whether more labeled data is likely to improve results

---

## Notes
- Keep the group-aware split strategy consistent across runs.
- Do not tune model settings on the held-out test set.
- Inspect image-mask alignment before training.
- Check qualitative predictions, not only numerical metrics.
- Save the selected threshold together with the final model checkpoint.
