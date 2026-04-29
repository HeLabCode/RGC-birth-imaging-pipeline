import numpy as np
from skimage.filters import gaussian
from skimage.morphology import disk, opening, closing, remove_small_holes


# ---------- Illumination mask ----------
def build_illumination_mask(stack, topk=3, radius_px=80, keep_frac=0.7, invert=False):
    """
    Compute a smooth illumination mask from the brightest Z-slices of a stack.

    The mask highlights well-lit regions based on Gaussian-smoothed intensity
    and a percentile threshold.

    Parameters
    ----------
    stack : list or ndarray
        Image stack (list of 2D slices or 3D array).
    topk : int
        Number of brightest slices to use for mask creation.
    radius_px : float
        Blur radius for Gaussian smoothing.
    keep_frac : float
        Intensity fraction threshold (≥ value is kept).
    invert : bool
        If True, invert the mask logic.

    Returns
    -------
    mask : ndarray (uint8)
        Binary mask of illuminated regions.
    field : ndarray
        The smoothed illumination field used to compute the mask.
    """
    p99 = np.array([np.percentile(s.astype(np.float32), 99) for s in stack])
    order = np.argsort(p99)[::-1][:topk]
    ref = np.median([stack[i].astype(np.float32) for i in order], axis=0)

    field = gaussian(ref, sigma=radius_px / 2, preserve_range=True)
    field /= np.median(field[field > 0])

    mask = field <= keep_frac if invert else field >= keep_frac
    mask = opening(mask, footprint=disk(3))
    mask = remove_small_holes(mask, max_size=4999)
    mask = closing(mask, footprint=disk(9))
    return mask.astype(np.uint8), field
