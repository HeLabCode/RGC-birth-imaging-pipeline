"""
Utility functions for FRET efficiency analysis.

This module contains helper functions for loading donor/FRET images,
loading and refining binary masks, separating cell instances, shrinking
cell masks, estimating local background, computing corrected FRET efficiency,
and saving quality-control visualization outputs.
"""

import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


# -------- Image loading utilities --------

def load_image_as_float(path):
    """
    Load a TIFF/image as a 2D float32 array.

    If RGB/RGBA, channels are averaged to grayscale.
    If a stack is provided, the first plane is used.
    """
    img = tifffile.imread(path)

    if img.ndim == 3 and img.shape[-1] in [3, 4]:
        img = img[..., :3].mean(axis=-1)

    elif img.ndim == 3:
        print(f"Warning: {path} appears to be a stack. Using the first plane.")
        img = img[0]

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image after loading, but got shape {img.shape}")

    return img.astype(np.float32)


def load_binary_mask(path, target_shape=None):
    """
    Load a binary mask.

    Pixels > 0 are treated as cell mask.
    If target_shape is provided, the mask must match the image shape.
    """
    mask = tifffile.imread(path)

    if mask.ndim == 3 and mask.shape[-1] in [3, 4]:
        mask = mask[..., :3].mean(axis=-1)

    elif mask.ndim == 3:
        print(f"Warning: {path} appears to be a stack. Using the first plane.")
        mask = mask[0]

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask after loading, but got shape {mask.shape}")

    mask = mask > 0

    if target_shape is not None and mask.shape != target_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {target_shape}. "
            "Please resize/register the mask before running this demo."
        )

    return mask.astype(np.uint8)


# -------- Display normalization utilities --------

def normalize_img(img, p_low=1, p_high=99):
    """
    Normalize an image to 0-1 for display using percentile contrast.
    """
    img = img.astype(np.float32)

    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)

    vmin, vmax = np.percentile(img[finite], (p_low, p_high))

    if vmax <= vmin:
        return np.zeros_like(img, dtype=np.float32)

    img_norm = (img - vmin) / (vmax - vmin)
    img_norm = np.clip(img_norm, 0, 1)

    return img_norm.astype(np.float32)


def normalize_for_display(img, p_low=1, p_high=99):
    """
    Alias for normalize_img(), used in preview functions.
    """
    return normalize_img(img, p_low=p_low, p_high=p_high)


# -------- Mask refinement and instance segmentation --------

def refine_to_instances(mask_bin, min_distance=8, min_size=40, peak_thresh=0.15):
    """
    Convert a binary cell mask into instance labels using distance transform
    and watershed.

    Parameters
    ----------
    mask_bin : 2D array
        Binary cell mask. Nonzero pixels are treated as cell pixels.
    min_distance : int
        Minimum distance between local maxima used as watershed seeds.
    min_size : int
        Minimum object size in pixels.
    peak_thresh : float
        Relative threshold for peak detection. The absolute threshold is
        peak_thresh * max(distance transform).

    Returns
    -------
    labels_clean : 2D int array
        Instance label image. 0 = background.
    """
    mask_bin = (mask_bin > 0)

    if mask_bin.sum() == 0:
        return np.zeros(mask_bin.shape, dtype=np.int32)

    mask_bin = morphology.remove_small_objects(mask_bin, max_size=min_size - 1)
    mask_bin = morphology.remove_small_holes(mask_bin, max_size=min_size - 1)
    mask_bin = morphology.closing(mask_bin, morphology.disk(1))

    if mask_bin.sum() == 0:
        return np.zeros(mask_bin.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(mask_bin)

    if distance.max() <= 0:
        return np.zeros(mask_bin.shape, dtype=np.int32)

    coords = peak_local_max(
        distance,
        labels=mask_bin,
        min_distance=min_distance,
        threshold_abs=peak_thresh * distance.max()
    )

    if len(coords) < 2:
        labels_simple = measure.label(mask_bin, connectivity=2)

        labels_clean = np.zeros_like(labels_simple, dtype=np.int32)
        new_label = 1

        for region in measure.regionprops(labels_simple):
            if region.area < min_size:
                continue

            labels_clean[labels_simple == region.label] = new_label
            new_label += 1

        return labels_clean.astype(np.int32)

    markers = np.zeros(mask_bin.shape, dtype=np.int32)

    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    labels = watershed(
        -distance,
        markers=markers,
        mask=mask_bin
    )

    labels_clean = np.zeros_like(labels, dtype=np.int32)
    new_label = 1

    for region in measure.regionprops(labels):
        if region.area < min_size:
            continue

        labels_clean[labels == region.label] = new_label
        new_label += 1

    return labels_clean.astype(np.int32)


def shrink_binary_mask(mask, shrink_factor=0.8):
    """
    Shrink a binary cell mask toward its centroid.

    This follows the original S3-style logic:
    each pixel coordinate is moved toward the centroid by shrink_factor.

    Parameters
    ----------
    mask : 2D bool array
        Binary mask for one cell.
    shrink_factor : float
        1.0 = no shrink.
        0.8 = shrink to 80% radial size.
        0.6 = stronger shrink.

    Returns
    -------
    new_mask : 2D bool array
        Shrunken cell mask.
    """
    mask = mask.astype(bool)
    coords = np.argwhere(mask)

    if coords.size == 0:
        return mask.copy()

    if shrink_factor >= 1:
        return mask.copy()

    if shrink_factor <= 0:
        raise ValueError("shrink_factor must be > 0.")

    centroid = coords.mean(axis=0)
    new_mask = np.zeros_like(mask, dtype=bool)

    for y, x in coords:
        vec = np.array([y, x]) - centroid
        new_yx = centroid + shrink_factor * vec
        ny, nx = np.round(new_yx).astype(int)

        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
            new_mask[ny, nx] = True

    new_mask &= mask
    new_mask = morphology.closing(new_mask, morphology.disk(1))
    new_mask &= mask

    return new_mask


# -------- Background subtraction utilities --------

def shrink_background_mask_from_cells(cell_mask, keep_fraction=0.80, min_size=20):
    """
    Create a background mask from non-cell pixels and keep only the background
    pixels farthest from cells.

    This removes background pixels immediately adjacent to cells, which may
    contain halo or cell-edge signal.

    Parameters
    ----------
    cell_mask : 2D bool array
        True = cell pixels.
    keep_fraction : float
        Fraction of non-cell pixels to keep.
        0.80 means remove the 20% of background pixels closest to cells.
    min_size : int
        Remove tiny background islands smaller than this size.

    Returns
    -------
    bg_mask_shrunk : 2D bool array
        Background mask used for background estimation.
    """
    if not (0 < keep_fraction <= 1):
        raise ValueError("keep_fraction must be in the range (0, 1].")

    cell_mask = cell_mask.astype(bool)
    background_mask = ~cell_mask

    if background_mask.sum() == 0:
        raise ValueError("No background pixels found. Cell mask covers the full image.")

    dist_to_cell = ndi.distance_transform_edt(background_mask)

    bg_dist_values = dist_to_cell[background_mask]

    if bg_dist_values.size == 0:
        raise ValueError("No valid background distance values found.")

    cutoff_percentile = (1.0 - keep_fraction) * 100.0
    distance_cutoff = np.percentile(bg_dist_values, cutoff_percentile)

    bg_mask_shrunk = background_mask & (dist_to_cell >= distance_cutoff)

    bg_mask_shrunk = morphology.remove_small_objects(
        bg_mask_shrunk.astype(bool),
        max_size=min_size - 1
    )

    if bg_mask_shrunk.sum() == 0:
        raise ValueError(
            "Shrunk background mask has no pixels. "
            "Try increasing keep_fraction or reducing min_size."
        )

    return bg_mask_shrunk.astype(bool)


def apply_background_subtraction(channel_img, bg_mask, percentile=50):
    """
    Subtract background from one channel.

    Parameters
    ----------
    channel_img : 2D array
        Raw intensity image.
    bg_mask : 2D bool array
        True = background pixels used for background estimation.
    percentile : float
        Percentile of background pixels to subtract.
        50 = median background.

    Returns
    -------
    bg_value : float
        Estimated background value.
    corrected_img : 2D float32 array
        Background-subtracted image. Negative values are set to zero.
    """
    bg_mask = bg_mask.astype(bool)

    bg_values = channel_img[bg_mask]
    bg_values = bg_values[np.isfinite(bg_values)]

    if bg_values.size == 0:
        bg_value = 0.0
    else:
        bg_value = float(np.percentile(bg_values, percentile))

    corrected_img = channel_img.astype(np.float32) - bg_value
    corrected_img[corrected_img < 0] = 0

    return bg_value, corrected_img.astype(np.float32)


def apply_background_subtraction_from_cell_mask(
    channel_img,
    cell_mask,
    keep_fraction=0.80,
    percentile=50,
    min_size=20
):
    """
    Estimate background from non-cell pixels farther away from cells and
    subtract it from one channel.

    Parameters
    ----------
    channel_img : 2D array
        Raw channel image.
    cell_mask : 2D bool array
        True = cell pixels.
    keep_fraction : float
        Fraction of non-cell background pixels farthest from cells to keep.
    percentile : float
        Percentile of selected background pixels used as background value.
    min_size : int
        Minimum size for background islands.

    Returns
    -------
    bg_value : float
        Estimated background value.
    corrected_img : 2D float32 array
        Background-subtracted image.
    bg_mask_shrunk : 2D bool array
        Background mask used for estimation.
    """
    bg_mask_shrunk = shrink_background_mask_from_cells(
        cell_mask=cell_mask,
        keep_fraction=keep_fraction,
        min_size=min_size
    )

    bg_value, corrected_img = apply_background_subtraction(
        channel_img=channel_img,
        bg_mask=bg_mask_shrunk,
        percentile=percentile
    )

    return bg_value, corrected_img, bg_mask_shrunk


# -------- FRET efficiency calculation --------

def calculate_fret_efficiency(
    I_Turq_raw,
    I_FRET_raw,
    A,
    B,
    C,
    D,
    QYd_QYa_ratio,
    Sd_Sa_ratio,
    Gd_Ga_ratio,
    clip_range=(-10, 10)
):
    """
    Calculate bleed-through-corrected FRET efficiency.

    Formula
    -------
    I_Turq = I_Turq_raw * C
    I_FRET = I_FRET_raw * D

    FRET_corr = (1 - A) * I_FRET - B * I_Turq

    D_lost = FRET_corr * QYd_QYa_ratio * Sd_Sa_ratio * Gd_Ga_ratio

    E_FRET = D_lost / (D_lost + I_Turq)

    Parameters
    ----------
    I_Turq_raw : 2D array
        Background-subtracted donor/Turquoise channel.
    I_FRET_raw : 2D array
        Background-subtracted FRET channel.
    A, B, C, D : float
        Correction constants.
    QYd_QYa_ratio, Sd_Sa_ratio, Gd_Ga_ratio : float
        Calibration constants.
    clip_range : tuple or None
        If not None, clip E_FRET to this range.

    Returns
    -------
    E_FRET : 2D float32 array
        Pixel-wise FRET efficiency.
    FRET_corr : 2D float32 array
        Bleed-through-corrected FRET signal.
    D_lost : 2D float32 array
        Estimated donor signal lost to FRET.
    """
    I_Turq = I_Turq_raw.astype(np.float32) * C
    I_FRET = I_FRET_raw.astype(np.float32) * D

    FRET_corr = (1 - A) * I_FRET - B * I_Turq

    D_lost = (
        FRET_corr
        * QYd_QYa_ratio
        * Sd_Sa_ratio
        * Gd_Ga_ratio
    )

    denominator = D_lost + I_Turq

    with np.errstate(divide="ignore", invalid="ignore"):
        E_FRET = D_lost / denominator

    E_FRET[~np.isfinite(E_FRET)] = np.nan
    
    if clip_range is not None:
        E_FRET = np.clip(E_FRET, clip_range[0], clip_range[1])

    return (
        E_FRET.astype(np.float32),
        FRET_corr.astype(np.float32),
        D_lost.astype(np.float32)
    )


# -------- Visualization utilities --------

def make_segmentation_preview(
    base_img,
    labels,
    output_path,
    cmap_name="tab20",
    linewidth=1.2,
    label_fontsize=7,
    label_color="white",
    label_outline_color="black",
    figsize=(6, 6),
    dpi=300
):
    """
    Save segmentation preview with differently colored cell outlines
    and cell ID numbers at cell centroids.

    Parameters
    ----------
    base_img : 2D array
        Raw image used as grayscale background.
    labels : 2D int array
        Labeled segmentation image. 0 = background.
    output_path : str
        Path to save preview image.
    cmap_name : str
        Matplotlib categorical colormap.
        Good options: 'tab20', 'hsv', 'gist_ncar'.
    linewidth : float
        Width of cell outline.
    label_fontsize : int
        Font size for cell ID numbers.
    """
    base_disp = normalize_for_display(base_img)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(base_disp, cmap="gray")

    cmap = plt.get_cmap(cmap_name)
    n_labels = int(labels.max())

    props = measure.regionprops(labels)

    for prop in props:
        label_id = prop.label

        cell_mask = labels == label_id
        boundary = segmentation.find_boundaries(cell_mask, mode="outer")

        color = cmap((label_id - 1) % cmap.N)

        ax.contour(
            boundary.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth
        )

        cy, cx = prop.centroid

        txt = ax.text(
            cx,
            cy,
            str(label_id),
            color=label_color,
            fontsize=label_fontsize,
            ha="center",
            va="center",
            weight="bold"
        )

        txt.set_path_effects([
            path_effects.Stroke(
                linewidth=1.5,
                foreground=label_outline_color
            ),
            path_effects.Normal()
        ])

    ax.set_title(f"Watershed segmentation: {n_labels} cells", fontsize=12)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


def make_background_preview(
    base_img,
    cell_mask,
    background_mask,
    output_path=None,
    figsize=(6, 6),
    dpi=300
):
    """
    Preview the background mask used for background subtraction.

    Cell mask is shown in red.
    Background pixels used for estimation are shown in cyan.
    """
    base_disp = normalize_for_display(base_img)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(base_disp, cmap="gray")

    ax.imshow(
        np.ma.masked_where(~background_mask, background_mask),
        cmap="cool",
        alpha=0.35
    )

    ax.imshow(
        np.ma.masked_where(~cell_mask, cell_mask),
        cmap="Reds",
        alpha=0.25
    )

    ax.set_title("Background mask used for subtraction")
    ax.axis("off")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close()


def save_per_cell_pseudocolor_and_labeled_colorbar(
    labels,
    values_by_label,
    pseudocolor_output_path,
    colorbar_output_path,
    vmin=0.6,
    vmax=1.0,
    cmap_name="coolwarm",
    colorbar_label="FRET efficiency",
    colorbar_ticks=None,
    pseudocolor_dpi=300,
    colorbar_dpi=300,
):
    """
    Save two separate TIFF images:
    1. Per-cell pseudocolor TIFF without colorbar
    2. Standalone labeled colorbar TIFF with tick labels
    """

    per_cell_img = np.full(labels.shape, np.nan, dtype=np.float32)

    for label_id, value in values_by_label.items():
        per_cell_img[labels == label_id] = value

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    rgba_img = cmap(norm(per_cell_img))
    bg_mask = np.isnan(per_cell_img)
    rgba_img[bg_mask, :] = [0, 0, 0, 1]

    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

    tifffile.imwrite(
        pseudocolor_output_path,
        rgb_img,
        photometric="rgb"
    )

    print(f"Saved pseudocolor TIFF to: {pseudocolor_output_path}")


    if colorbar_ticks is None:
        colorbar_ticks = np.linspace(vmin, vmax, 5)

    fig, ax = plt.subplots(figsize=(1.5, 3), dpi=colorbar_dpi)

    gradient = np.linspace(vmax, vmin, 512).reshape(512, 1)

    ax.imshow(
        gradient,
        cmap=cmap,
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xticks([])
    ax.yaxis.tick_right()

    tick_positions = [
        (vmax - t) / (vmax - vmin) * (512 - 1)
        for t in colorbar_ticks
    ]

    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{t:.2f}" for t in colorbar_ticks], fontsize=12)

    ax.set_ylabel(
        colorbar_label,
        fontsize=14,
        rotation=270,
        labelpad=25
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    plt.tight_layout()

    fig.canvas.draw()

    colorbar_img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    colorbar_img = colorbar_img.astype(np.uint8)

    tifffile.imwrite(
        colorbar_output_path,
        colorbar_img,
        photometric="rgb"
    )
    plt.close()

    print(f"Saved labeled colorbar TIFF to: {colorbar_output_path}")

    return per_cell_img, rgb_img, colorbar_img
