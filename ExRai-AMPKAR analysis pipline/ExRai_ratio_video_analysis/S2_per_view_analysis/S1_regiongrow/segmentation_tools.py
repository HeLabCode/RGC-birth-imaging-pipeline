import cv2
import numpy as np
from skimage import morphology
from skimage.segmentation import watershed
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="skimage")


# -------- Adaptive region growing segmentation  --------

def adaptive_region_grow(img, seed_rc, rel_drop=0.6, max_radius=25, mask_existing=None):

    """
    Segment a single cell from a user-provided seed using adaptive region growing.

    The seed is first refined by locating the centroid of the brightest local
    core in a small neighborhood around the clicked point. Local intensity
    statistics from this neighborhood are then used to define a growth
    tolerance that combines relative intensity drop, local variability, and
    local dynamic range. The region is expanded iteratively within a circular
    search area, while preventing overlap with already segmented cells when
    an existing mask is provided. Final masks are post-processed by binary
    closing, hole filling, and small-object removal.

    Parameters
    ----------
    img : ndarray
        Input 2D image slice.
    seed_rc : tuple of int
        Seed coordinates as (row, col).
    rel_drop : float, optional
        Relative intensity drop allowed from the seed value during growth.
    max_radius : int, optional
        Maximum radius of the circular search region around the seed.
    mask_existing : ndarray, optional
        Existing segmentation mask used to prevent overlap with previously
        labeled regions.

    Returns
    -------
    ndarray
        Binary mask of the segmented region as uint8.
    r0, c0 = seed_rc
    """
    r0, c0 = seed_rc

    
    img = cv2.GaussianBlur(img.astype(np.float32), (3, 3), 0)
    h, w = img.shape

    r0 = int(np.clip(r0, 0, h - 1))
    c0 = int(np.clip(c0, 0, w - 1))

    R_SEED = 7
    r1, r2 = max(0, r0 - R_SEED), min(h, r0 + R_SEED + 1)
    c1, c2 = max(0, c0 - R_SEED), min(w, c0 + R_SEED + 1)
    seed_patch = img[r1:r2, c1:c2]

    if seed_patch.size == 0:
        return np.zeros_like(img, np.uint8)

    core_thresh = np.percentile(seed_patch, 80)
    core_mask = seed_patch >= core_thresh

    if not np.any(core_mask):
        core_mask = np.ones_like(seed_patch, bool)

    core_coords = np.column_stack(np.nonzero(core_mask))
    dy, dx = core_coords.mean(axis=0)
    r0 = int(r1 + dy)
    c0 = int(c1 + dx)

    r0 = int(np.clip(r0, 0, h - 1))
    c0 = int(np.clip(c0, 0, w - 1))

    r1, r2 = max(0, r0 - R_SEED), min(h, r0 + R_SEED + 1)
    c1, c2 = max(0, c0 - R_SEED), min(w, c0 + R_SEED + 1)
    seed_patch = img[r1:r2, c1:c2]

    if seed_patch.size == 0:
        return np.zeros_like(img, np.uint8)

    p20 = np.percentile(seed_patch, 20)
    p80 = np.percentile(seed_patch, 80)
    seed_val = p80
    local_std = np.std(seed_patch)
    bg_val = np.median(seed_patch)
    patch_range = max(p80 - p20, 1e-3)

    if seed_val > bg_val + 3 * local_std:
        seed_val = bg_val + 3 * local_std

    brightness_factor = np.clip(seed_val / 60.0, 0.5, 2.0)


    tol_rel = seed_val * (1 - rel_drop) 
    tol_std = local_std * 1.5
    tol_rng = 0.3 * patch_range
    tol = max(tol_rel, tol_std, tol_rng, 5.0) # tolerance: mix relative, std, and local range

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - r0) ** 2 + (X - c0) ** 2)
    circular_mask = dist <= max_radius

    region = np.zeros_like(img, bool)
    initial_mask = dist <= 2
    region = initial_mask & circular_mask & (np.abs(img - seed_val) <= tol)
    if not np.any(region):
        region[r0, c0] = True

    region_mean = img[region].mean()
    current_tol = tol

    extreme_seed = seed_val > bg_val + 4 * local_std


    base_ct = 0.80 - (brightness_factor * 0.10)
    contrast_threshold = float(np.clip(base_ct, 0.55, 0.80))

    while True:
        dil = morphology.dilation(region)
        candidates = np.logical_and(dil, ~region)
        candidates &= circular_mask
        if mask_existing is not None:
            candidates &= (mask_existing == 0)

        diff = np.abs(img - region_mean)
        newpix = candidates & (diff <= current_tol)

        if np.sum(region) == 1:
            newpix = candidates.copy()

        if not np.any(newpix):
            break

        if not extreme_seed:
            if img[newpix].mean() < seed_val * contrast_threshold:
                break

        region |= newpix
        region_mean = img[region].mean()

        decay_rate = 0.90 + (brightness_factor * 0.02)
        current_tol *= decay_rate

        if np.count_nonzero(region) > np.pi * max_radius ** 2:
            break

    region = morphology.closing(region, morphology.disk(2))
    region = morphology.remove_small_holes(region, max_size=39)
    region = morphology.remove_small_objects(region, max_size=19)
    return region.astype(np.uint8)



# --------  Watershed-based mask splitting  --------

def watershed_split(mask, seed1, seed2):
    """
    Split a binary mask into two regions using watershed segmentation.

    Two seed points define the initial markers for watershed segmentation,
    which divides the mask into two separate subregions along the boundary
    of maximal distance transform gradient.

    Parameters
    ----------
    mask : ndarray (bool or uint8)
        Binary mask of the region to be split.
    seed1, seed2 : tuple of int
        Pixel coordinates (row, col) of the two seed points.

    Returns
    -------
    ws : ndarray (int)
        Label map where 1 and 2 correspond to the two separated regions.
    contour : None
        Placeholder (for compatibility with previous interface).
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    markers = np.zeros_like(mask, np.int32)
    markers[seed1] = 1
    markers[seed2] = 2
    ws = watershed(-dist, markers, mask=mask)
    return ws, None
