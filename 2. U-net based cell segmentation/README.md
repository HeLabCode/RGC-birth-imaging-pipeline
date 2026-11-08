# U-Net Cell Segmentation Pipeline

## Overview

This module trains, optimizes, evaluates, and applies a supervised U-Net model for segmenting cells in two-photon fluorescence microscopy images. The training workflow uses 100 paired input images and manually annotated binary masks, group-aware train/validation/test splitting, a ResNet-50 encoder, and hyperparameter optimization. A separate notebook uses a saved checkpoint to predict binary masks for new images.

**Two distinct workflows are included:** (1) training/evaluating the model and (2) running inference with the supplied model. Using a supplied checkpoint is separate from retraining and hyperparameter optimization.

## Repository organization

The actual folder names in this archive begin with `2.1` through `2.4`. Keep those names unchanged when following these instructions.

| Path | Contents |
| --- | --- |
| [`2.1 Neural Network Set Up/Example images/`](2.1%20Neural%20Network%20Set%20Up/Example%20images/) | 100 microscopy TIFF images (e.g., `img_211.tif`) |
| [`2.1 Neural Network Set Up/Example images manual mask/`](2.1%20Neural%20Network%20Set%20Up/Example%20images%20manual%20mask/) | 100 manually drawn masks (e.g., `img_211_mask.tif`) |
| [`2.1 Neural Network Set Up/images_similarity.csv`](2.1%20Neural%20Network%20Set%20Up/images_similarity.csv) | Image grouping information for group-aware splitting |
| [`2.2 Selection for best NN parameters/U_net_train_model_hparam_search.ipynb.ipynb`](2.2%20Selection%20for%20best%20NN%20parameters/U_net_train_model_hparam_search.ipynb.ipynb) | Training, hyperparameter optimization, model selection, and evaluation notebook |
| [`2.2 Selection for best NN parameters/semantic_segmentation_output/`](2.2%20Selection%20for%20best%20NN%20parameters/semantic_segmentation_output/) | Saved model, split metadata, best hyperparameters, metrics, training history and plots |
| [`2.3 Running Neural Network/NN_binary_mask.ipynb`](2.3%20Running%20Neural%20Network/NN_binary_mask.ipynb) | Notebook for applying the pretrained model to an image or image folder |
| [`2.3 Running Neural Network/best_semantic_model.pth`](2.3%20Running%20Neural%20Network/best_semantic_model.pth) | Supplied model checkpoint for inference |
| [`2.3 Running Neural Network/Example images and mask/`](2.3%20Running%20Neural%20Network/Example%20images%20and%20mask/) | One sample raw image, saved neural-network mask, and manual mask |
| [`2.4 Neural Network evaluation/`](2.4%20Neural%20Network%20evaluation/) | Two raw images, two manual labels, and two neural-network mask examples |

The training notebook filename ends in `.ipynb.ipynb`, as shown above.

## Example input data

| Attribute | Included data |
| --- | --- |
| Modality | Two-photon fluorescence microscopy images used for cell segmentation |
| Training input images | 100 TIFFs in `2.1 Neural Network Set Up/Example images/` |
| Reference labels | 100 matching TIFF masks in `Example images manual mask/` |
| Training naming convention | `img_211.tif` → `img_211_mask.tif` |
| Representative training image size | 512 × 512 pixels, 8-bit TIFF |
| Saved split | 73 train, 15 validation, 12 test images (`split_metadata.json`) |
| Group definition | `images_similarity.csv`, which includes an image ID and `group_id` |
| Inference demonstration | `raw_image.tif`, `NN_binary_mask.tif`, `manual_binary_mask.tif` in the `2.3` example folder |
| Inference sample size | 300 × 300 pixels, 8-bit TIFF |
| Further visual comparisons | `example1_*` and `example2_*` TIFFs in `2.4 Neural Network evaluation/` |

The TIFFs are analysis inputs and example masks. The folder does not establish whether the images are unmodified microscope acquisition files.

## Requirements

The training workflow imports Python, Jupyter, PyTorch, Segmentation Models PyTorch, Albumentations, Optuna, scikit-learn, OpenCV, TIFF utilities, and supporting analysis packages. Install in a separate environment:

```bash
pip install jupyterlab numpy pandas matplotlib opencv-python torch scikit-learn albumentations segmentation-models-pytorch optuna natsort tifffile scipy
```

The inference notebook also imports `scikit-image`. Compatibility with all newer package versions is not established. Results can depend on software versions, hardware, and random seeds.

## Inference with the supplied model

- [ ] Retain the directory `2.3 Running Neural Network/` with `NN_binary_mask.ipynb`, `best_semantic_model.pth`, and `Example images and mask/`.
- [ ] Start Jupyter **from `2.3 Running Neural Network/`** and open [`NN_binary_mask.ipynb`](2.3%20Running%20Neural%20Network/NN_binary_mask.ipynb).
- [ ] In the configuration cell, confirm `INPUT_IMAGE = "Example images and mask/raw_image.tif"`, `MODEL_PATH = "best_semantic_model.pth"`, and `OUT_DIR = "Example images and mask"`.
- [ ] Confirm the `resnet50` encoder, model architecture, preprocessing, and `THRESHOLD = 0.75` match the intended checkpoint.
- [ ] The supplied 300 × 300 example may not run unchanged with this U-Net, which was trained using 512 × 512 images (see limitations below).
- [ ] Run the notebook and inspect the predicted mask against `manual_binary_mask.tif` and the previously provided `NN_binary_mask.tif`.
- [ ] Check the output `Example images and mask/NN_binary_mask.tif`; running the notebook overwrites this existing file.

**Implementation limitations:**

1. The training model uses a U-Net with a ResNet-50 encoder and 512 × 512 training images. The provided inference example is **300 × 300**, and the inference notebook does not pad or resize the image before calling the model. Encoder/decoder spatial dimensions may therefore be incompatible. The supplied inference notebook has **not been verified to run end-to-end unchanged** with that example. 
2. The inference notebook saves **every input mask as `NN_binary_mask.tif`**, even if `INPUT_IMAGE` is changed to a folder. In batch mode, later images overwrite earlier masks. 
3. The inference notebook hardcodes the threshold to `0.75` instead of automatically reading the checkpoint's `best_threshold`. This value may differ from the threshold selected during validation.
4. The pretrained checkpoint is approximately **130 MB**. It is larger than GitHub's normal **100 MiB per-file Git limit**; ordinary Git cannot store it as a regular file on GitHub. The file is included in this downloaded ZIP.

These warnings describe the actual code; **the notebook and model were not modified** as part of this README update.

## Training and model evaluation

- [ ] Keep the `2.1` training image and mask folders, `images_similarity.csv`, and the `2.2` notebook together under the module root.
- [ ] Open `2.2 Selection for best NN parameters/U_net_train_model_hparam_search.ipynb.ipynb` with the Jupyter working directory set to **`2.2 Selection for best NN parameters/`**.
- [ ] The training notebook uses `../1. Neural Network Set Up/...` in its configuration, but the supplied directory is `../2.1 Neural Network Set Up/...`. The configured `Img_dir`, `mask_dir`, and `groups_csv_path` therefore do not resolve without corresponding path adjustments. The notebook has not been changed.
- [ ] Confirm training settings (`seed = 28`, `input_size = 512`, `n_trials = 70`) and inspect group-aware splitting. The saved split metadata identifies 73/15/12 images in train/validation/test; the group-aware process, not a fixed image-count constraint, produces the split.
- [ ] Check the loaded image/mask pairs and the augmentation preview before training.
- [ ] Run the full training and hyperparameter search only if you intend to retrain. It can be time-consuming and uses pretrained encoder weights, which may need to download on first use.
- [ ] Inspect `semantic_segmentation_output/selected_best_hyperparameters.json`, `split_metadata.json`, `test_summary_metrics.csv`, and `per_image_test_metrics.csv` to compare the new run with the archived model evaluation.
- [ ] The saved checkpoint provides an inference path separate from training.

The archived hyperparameter-search folder contains summaries and one selected trial, not saved artifacts for every trial.

## Workflow

1. **Prepare labeled training inputs:** load paired fluorescence TIFFs and manually created binary segmentation masks.
2. **Group related fields of view:** use `images_similarity.csv` and group-aware splitting to reduce leakage across training, validation, and held-out test data.
3. **Preprocess images and augment training data:** normalize, resize to the configured training size, and apply the training transformations.
4. **Build the U-Net:** ResNet-50 encoder with ImageNet initialization and a single-channel binary segmentation output.
5. **Train in two stages:** first freeze the encoder, then fine-tune the complete network with weighted BCE + Dice loss.
6. **Optimize parameters:** evaluate Optuna-selected configurations using validation data and save the selected trial.
7. **Select the binary threshold:** compare segmentation thresholds using validation data, without tuning on the held-out test set.
8. **Evaluate:** compute test Dice, IoU, precision, recall, and other classification/segmentation metrics; save per-image and summary results.
9. **Run inference separately:** load a trained checkpoint and create a binary segmentation mask from a fluorescence image.

## Key configuration values

| Parameter | Value in supplied notebooks | Context |
| --- | --- | --- |
| `seed` | `28` | Training and splitting |
| `input_size` | `512` | Training preprocessing |
| Encoder | `resnet50` | Training and inference |
| Encoder initialization | `imagenet` | Training |
| Train / validation / test target fractions | `0.70 / 0.15 / 0.15` | Group-aware splitting |
| `n_trials` | `70` | Optuna maximum search count |
| Search timeout | `1.5 * 3600` seconds | Optuna search duration cap |
| Training model output | One-channel binary segmentation | U-Net |
| Inference `THRESHOLD` | `0.75` | Hardcoded in inference notebook |
| Best training hyperparameters | `selected_best_hyperparameters.json` | Exact selected configuration |
| Final test split | `split_metadata.json` | Archived image IDs and groups |

Selected hyperparameters and trial results are included under `semantic_segmentation_output/`.

## Expected outputs

| File or folder | Description |
| --- | --- |
| `2.2 .../semantic_segmentation_output/best_semantic_model.pth` | Final saved model checkpoint |
| `2.2 .../semantic_segmentation_output/split_metadata.json` | Image/group assignments for training, validation, and test |
| `2.2 .../semantic_segmentation_output/selected_best_hyperparameters.json` | Selected training configuration |
| `2.2 .../semantic_segmentation_output/training_history.csv` | Saved training history |
| `2.2 .../semantic_segmentation_output/validation_threshold_search.csv` | Validation threshold search results |
| `2.2 .../semantic_segmentation_output/test_summary_metrics.csv` | Test-set summary results |
| `2.2 .../semantic_segmentation_output/per_image_test_metrics.csv` | Individual test-image measurements |
| `2.2 .../semantic_segmentation_output/hyperparameter_search/` | Trial and data-efficiency summaries, logs, and selected run artifacts |
| `2.3 .../Example images and mask/NN_binary_mask.tif` | Preprovided sample binary segmentation prediction; overwritten by inference |
| `2.4 Neural Network evaluation/` | Example comparisons of images, manual masks, and predicted masks |

## Notes

- Folder names and links correspond to the supplied archive.
- The notebooks retain the path and output-naming limitations described above.
- Running a new optimization search can produce different selected configurations and metrics from those in the archived outputs, even with controlled random seeds.
- The included model checkpoints exceed the standard GitHub file-size limit for ordinary Git.
