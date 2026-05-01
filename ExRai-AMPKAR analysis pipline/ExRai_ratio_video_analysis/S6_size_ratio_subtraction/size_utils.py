import numpy as np
import pandas as pd
from skimage import io



# ---------- Higest slice intensity ----------

def get_best_slice_from_maskstack(path, raw_stack=None):
    """
    Return mask, mean intensity, and Z-index of the slice 
    with the highest mean intensity inside the mask.

    Parameters
    ----------
    path : str
        Path to the per-track mask stack (e.g. *_maskstack.tif).
    raw_stack : ndarray, optional
        Corresponding raw image stack (same Z-order). 
        If None, intensity is computed directly from mask binary area.

    Returns
    -------
    best_mask : ndarray
        Binary mask of the slice with the highest mean intensity.
    best_val : float
        Mean intensity value inside that mask.
    best_z : int
        Index of the Z-slice with the highest mean intensity.
    """
    mstk = io.imread(path)
    if mstk.ndim == 2:
        mstk = mstk[np.newaxis, ...]

    best_val, best_mask, best_z = -np.inf, None, 0

    for zi in range(mstk.shape[0]):
        mask = (mstk[zi] > 0).astype(bool)
        if not np.any(mask):
            continue

        if raw_stack is not None and zi < raw_stack.shape[0]:
            intensity = float(raw_stack[zi][mask].mean())
        else:
            intensity = mask.sum()

        if intensity > best_val:
            best_val = intensity
            best_mask = mask.astype(np.uint8)
            best_z = zi

    return best_mask, best_val, best_z



# ---------- Collect best slice info ----------

def collect_best_slices(mask_map, raw_stack=None):
    """
    For each tracked cell, identify the Z-slice with the highest mean intensity.

    For every mask stack in `mask_map`, this function calls
    `get_best_slice_from_maskstack()` to identify the slice where the
    cell shows the highest mean intensity (optionally using the provided
    raw image stack).

    Parameters
    ----------
    mask_map : dict
        Dictionary mapping track IDs → mask stack file paths.
    raw_stack : ndarray, optional
        Corresponding raw image stack (same Z-order as masks).
        If None, selection is based on mask area instead of intensity.

    Returns
    -------
    dict
        Mapping {track_id: (best_value, best_z)} where:
            - best_value : float, mean intensity (or area if no raw_stack)
            - best_z     : int, Z index of the slice with that maximum
    """
    out = {}
    for tid, path in mask_map.items():
        _, best_val, z_best = get_best_slice_from_maskstack(path, raw_stack)
        out[tid] = (best_val, z_best)
    return out



# ----------  Compute area ratios ----------

def compute_area_ratios(mapping_csv, mask_map_800, mask_map_920):
    """
    Compute area ratios between matched 800 nm and 920 nm cells.

    Uses the provided cross-channel mapping and the best-slice information
    from each mask stack (via `collect_best_slices`). For each mapped pair,
    it computes the ratio between the larger and smaller area/intensity
    as a symmetric measure of size difference.

    Parameters
    ----------
    mapping_csv : str
        Path to the CSV file containing the 800→920 mapping.
    mask_map_800, mask_map_920 : dict
        Dictionaries mapping track IDs → mask stack file paths
        for the 800 nm and 920 nm channels.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - track_800, track_920 : paired cell IDs
        - area800, area920     : measured areas (or intensities)
        - ratio_symmetric      : larger/smaller ratio
        - z800_best, z920_best : Z-indices of best slices
    """
    df = pd.read_csv(mapping_csv)
    mapping = dict(zip(df.track_800, df.track_920))

    best800 = collect_best_slices(mask_map_800)
    best920 = collect_best_slices(mask_map_920)

    rows = []
    for t800, t920 in mapping.items():
        if t800 not in best800 or t920 not in best920:
            continue
        area800, z800 = best800[t800]
        area920, z920 = best920[t920]
        ratio = max(area800, area920) / min(area800, area920) if (area800 > 0 and area920 > 0) else np.nan
        rows.append({
            "track_800": t800,
            "track_920": t920,
            "area800": area800,
            "area920": area920,
            "ratio_symmetric": ratio,
            "z800_best": z800,
            "z920_best": z920
        })
    return pd.DataFrame(rows)



