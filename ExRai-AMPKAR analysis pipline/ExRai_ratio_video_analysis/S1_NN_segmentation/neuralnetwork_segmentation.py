import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import torch

from ipywidgets import IntSlider, interact
from scipy import ndimage as ndi
from segmentation_models_pytorch import Unet
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# -------- Image normalization and NN prediction --------

def nn_normalize_img(img):
    """
    Normalize a raw image slice to the [0, 1] range using robust percentiles.

    Intensities are scaled between the 1st and 99th percentiles to reduce the
    influence of extreme outliers before neural-network inference.

    Parameters
    ----------
    img : ndarray
        Input 2D image.

    Returns
    -------
    ndarray
        Normalized image in float32 format, clipped to [0, 1].
    """
    img = img.astype(np.float32)
    vmin, vmax = np.percentile(img, (1, 99))
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    return np.clip(img, 0, 1)


def nn_predict_mask(img, nn_model, device):
    """
    Run neural-network inference on a single image slice.

    The grayscale image is normalized, replicated into three channels to match
    the network input format, and passed through the model. The output
    probability map is thresholded to obtain a binary segmentation mask.

    Parameters
    ----------
    img : ndarray
        Input 2D image.
    nn_model : torch.nn.Module
        Loaded segmentation model.
    device : torch.device
        Device used for inference.

    Returns
    -------
    prob : ndarray
        Predicted probability map.
    mask : ndarray
        Binary mask obtained by thresholding the probability map.
    """
    img_n = nn_normalize_img(img)
    img_rgb = np.repeat(img_n[..., None], 3, axis=2)
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        out = nn_model(tensor)
        if isinstance(out, tuple):
            out = out[0]
        prob = torch.sigmoid(out).cpu().numpy()[0, 0]

    mask = (prob > 0.5).astype(np.uint8)
    return prob, mask


# -------- Instance refinement from binary NN mask --------

def nn_refine_to_instances(mask_bin, min_distance=8, min_size=40, peak_thresh=0.15):
    """
    Convert a binary neural-network mask into individual cell instances.

    A distance transform is computed on the binary mask, local maxima are used
    as watershed markers, and small regions are removed after segmentation.
    If fewer than two peaks are detected, connected-component labeling is used
    as a simpler fallback.

    Parameters
    ----------
    mask_bin : ndarray
        Binary segmentation mask.
    min_distance : int, optional
        Minimum distance between local maxima used as watershed seeds.
    min_size : int, optional
        Minimum region area retained after instance refinement.
    peak_thresh : float, optional
        Relative threshold for local-maximum detection on the distance map.

    Returns
    -------
    ndarray
        Integer label image containing separated cell instances.
    """
    mask_bin = (mask_bin > 0).astype(np.uint8)

    if mask_bin.sum() == 0:
        return np.zeros_like(mask_bin, dtype=np.int32)

    distance = ndi.distance_transform_edt(mask_bin)

    coords = peak_local_max(
        distance,
        labels=mask_bin,
        min_distance=min_distance,
        threshold_abs=peak_thresh * distance.max()
    )

    if len(coords) < 2:
        labels_simple = measure.label(mask_bin, connectivity=2)
        for region in measure.regionprops(labels_simple):
            if region.area < min_size:
                labels_simple[labels_simple == region.label] = 0
        return labels_simple.astype(np.int32)

    markers = np.zeros_like(mask_bin, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i

    labels = watershed(-distance, markers, mask=mask_bin)

    for region in measure.regionprops(labels):
        if region.area < min_size:
            labels[labels == region.label] = 0

    return labels.astype(np.int32)


# -------- Model loading --------

def _load_model(model_path):
    """
    Load the pretrained U-Net model used for cell segmentation.

    Parameters
    ----------
    model_path : str or Path
        Path to the trained `.pth` model weights.

    Returns
    -------
    nn_model : torch.nn.Module
        Loaded segmentation model in evaluation mode.
    device : torch.device
        Device selected for inference.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    nn_model = Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    nn_model.load_state_dict(torch.load(model_path, map_location=device))
    nn_model.to(device).eval()

    return nn_model, device


# -------- Main NN segmentation routine --------

def run_S11_nn(channel, stack800, z800, stack920, z920, S11_8, S11_9, model_path):
    """
    Run neural-network segmentation on all slices of one excitation channel.

    For each Z-slice, the function performs neural-network prediction,
    refines the binary mask into individual cell instances, and saves both
    the instance labels (`.npy`) and binary masks (`.tif`) to disk.

    Parameters
    ----------
    channel : str
        Channel to process: "800" or "920".
    stack800, z800, stack920, z920
        Loaded image stacks and corresponding Z indices.
    S11_8, S11_9 : str
        Output folders for 800 nm and 920 nm NN results.
    model_path : str or Path
        Path to the trained `.pth` model.
    """
    print(f"S11: Running NN segmentation for channel {channel}...")

    nn_model, device = _load_model(model_path)

    if channel == "800":
        stack = stack800
        z_list = z800
        out_dir = S11_8
        tag = "S11_800"
    elif channel == "920":
        stack = stack920
        z_list = z920
        out_dir = S11_9
        tag = "S11_920"
    else:
        raise ValueError("channel must be '800' or '920'")

    os.makedirs(out_dir, exist_ok=True)

    for img, z in zip(stack, z_list):
        prob, mask_bin = nn_predict_mask(img, nn_model, device)
        labels = nn_refine_to_instances(mask_bin)

        out_npy = os.path.join(out_dir, f"Z{z:03d}_nn_labels.npy")
        out_tif = os.path.join(out_dir, f"Z{z:03d}_nn_mask.tif")

        np.save(out_npy, labels)
        tifffile.imwrite(out_tif, (labels > 0).astype(np.uint8))

        print(f"[{tag}] Z{z:03d} → {labels.max()} cells")

    print(f"✅ S11 NN segmentation complete for channel {channel}.")


# -------- Interactive viewer for QC --------

def view_S11_nn_results(channel, stack800, z800, stack920, z920, S11_8, S11_9):
    """
    Display neural-network segmentation results for visual quality control.

    The viewer shows, for each selected Z-slice, the raw image, the binary
    neural-network mask, and the final instance-labeled segmentation.

    Parameters
    ----------
    channel : str
        Channel to visualize: "800" or "920".
    stack800, z800, stack920, z920
        Loaded image stacks and corresponding Z indices.
    S11_8, S11_9 : str
        Output folders containing saved NN segmentation results.
    """
    if channel == "800":
        stack = stack800
        z_list = z800
        nn_dir = S11_8
        title_ch = "800 nm"
    elif channel == "920":
        stack = stack920
        z_list = z920
        nn_dir = S11_9
        title_ch = "920 nm"
    else:
        raise ValueError("channel must be '800' or '920'")

    def _show(i):
        z = z_list[i]
        raw = stack[i]

        labels = np.load(os.path.join(nn_dir, f"Z{z:03d}_nn_labels.npy"))
        mask = (labels > 0).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(raw, cmap="gray")
        ax[0].set_title(f"{title_ch}  RAW  |  Z={z}")
        ax[0].axis("off")

        ax[1].imshow(mask, cmap="gray")
        ax[1].set_title("NN Binary Mask")
        ax[1].axis("off")

        ax[2].imshow(labels, cmap="nipy_spectral")
        ax[2].set_title(f"NN Instances (n={labels.max()})")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

    slider = IntSlider(
        min=0,
        max=len(z_list) - 1,
        step=1,
        value=0,
        description=f"{title_ch} Z-index:"
    )

    interact(_show, i=slider)
