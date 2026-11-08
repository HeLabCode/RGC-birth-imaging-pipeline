
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, measure
from collections import defaultdict, Counter

from common_functions.data_utils import index_maskstacks, load_raw_map



# ----------  Hungarian (optimal assignment) ----------

def _hungarian_match(c8: pd.DataFrame, c9: pd.DataFrame, max_dist: float = 30) -> dict:
    """
    Perform Hungarian (optimal assignment) matching between two centroid sets.

    Parameters
    ----------
    c8 : pandas.DataFrame
        Centroids from 800 nm channel, with columns ['track_id', 'x_ref', 'y_ref'].
    c9 : pandas.DataFrame
        Centroids from 920 nm channel, with columns ['track_id', 'x_ref', 'y_ref'].
    max_dist : float, optional
        Maximum allowed Euclidean distance for a valid match (default: 30 px).

    Returns
    -------
    dict
        Mapping {track_id_800: track_id_920} for matched centroids.
    """
    if c8.empty or c9.empty:
        return {}
    ids800 = c8["track_id"].values
    ids920 = c9["track_id"].values
    D = distance_matrix(c8[["x_ref", "y_ref"]].values,
                        c9[["x_ref", "y_ref"]].values)
    row_ind, col_ind = linear_sum_assignment(D)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if D[r, c] <= max_dist:
            mapping[int(ids800[r])] = int(ids920[c])
    return mapping


# ----------  Nearest-neighbor ----------

def _nearest_neighbor_match(c8: pd.DataFrame, c9: pd.DataFrame, max_dist: float = 30) -> dict:
    """
    Perform nearest-neighbor matching between two centroid sets.

    For each centroid in channel 800, the nearest centroid in channel 920
    is assigned if within the maximum distance threshold.

    Parameters
    ----------
    c8 : pandas.DataFrame
        Centroids from 800 nm channel, with columns ['track_id', 'x_ref', 'y_ref'].
    c9 : pandas.DataFrame
        Centroids from 920 nm channel, with columns ['track_id', 'x_ref', 'y_ref'].
    max_dist : float, optional
        Maximum allowed Euclidean distance for a valid match (default: 30 px).

    Returns
    -------
    dict
        Mapping {track_id_800: track_id_920} for matched centroids.
    """
    if c8.empty or c9.empty:
        return {}
    ids800 = c8["track_id"].values
    ids920 = c9["track_id"].values
    D = distance_matrix(c8[["x_ref", "y_ref"]].values,
                        c9[["x_ref", "y_ref"]].values)
    mapping = {}
    for i, drow in enumerate(D):
        min_j = np.argmin(drow)
        if drow[min_j] <= max_dist:
            mapping[int(ids800[i])] = int(ids920[min_j])
    return mapping



# ---------- Centroid extraction ----------

def compute_centroids_from_masks(channel: str, z: int, VIEW_DIR, S2_8, S2_9) -> pd.DataFrame:
    """
    Extract object centroids for a specific Z-slice from tracked mask stacks.

    Automatically handles 0-based vs 1-based Z indexing between raw images
    and mask stacks. If no centroid is found, returns an empty DataFrame
    with the correct columns for downstream safety.

    Parameters
    ----------
    channel : str
        Channel identifier ("800" or "920").
    z : int
        Target Z index (from raw image list).
    VIEW_DIR : str
        Root folder containing raw Z-slices.
    S2_8, S2_9 : str
        Paths to the S2 segmentation outputs.

    Returns
    -------
    pandas.DataFrame
        Columns: ['track_id', 'x_ref', 'y_ref', 'z']
    """
    mask_map = index_maskstacks(channel, S2_8, S2_9)
    zmap, z_list, _ = load_raw_map(channel, VIEW_DIR)
    if not z_list:
        print(f"⚠️ No raw slices found for channel {channel} under {VIEW_DIR}")
    rows = []

    for tid, path in mask_map.items():
        try:
            mstk = io.imread(path)
        except Exception:
            continue
        if mstk.ndim == 2:
            mstk = mstk[np.newaxis, ...]

        if z not in z_list:
            continue
        zi = z_list.index(z)

        if zi >= mstk.shape[0]:
            if zi - 1 < mstk.shape[0]:
                zi -= 1
            else:
                continue

        mask = mstk[zi] > 0
        if not np.any(mask):
            continue

        props = measure.regionprops(mask.astype(np.uint8))
        if props:
            y, x = props[0].centroid
            rows.append({
                "track_id": int(tid),
                "x_ref": float(x),
                "y_ref": float(y),
                "z": int(z)
            })

    if not rows:
        return pd.DataFrame(columns=["track_id", "x_ref", "y_ref", "z"])
    return pd.DataFrame(rows)




# ---------- Slice-by-slice centroid comparison ----------

def compare_centroids_z(z800: int, z920: int,
                        raw800_map, raw920_map, VIEW_DIR, S2_8, S2_9,
                        max_dist: float = 30,
                        offset_x: float = 0.0, offset_y: float = 0.0,
                        method: str = "Hungarian") -> dict:
    """
    Compare centroids between two Z-slices from the 800 nm and 920 nm stacks.

    Compare and match centroids between one 800 nm slice and one 920 nm slice.

    For a chosen pair of Z-slices, this function retrieves all cell centroids
    from both channels, applies a geometric offset if needed, and computes
    one-to-one matches between the two sets using either the Hungarian or
    Nearest-Neighbor algorithm.

    It also generates a visualization showing matched points connected by lines.

    Parameters
    ----------
    z800, z920 : int
        Z-slice indices for 800 nm and 920 nm stacks.
    raw800_map, raw920_map : dict
        Mapping from Z index → file path for raw images.
    VIEW_DIR : str
        Root directory containing image data.
    S2_8, S2_9 : str
        Directories containing S2 segmentation results for 800 and 920 channels.
    max_dist : float
        Maximum distance (pixels) for matching centroids.
    offset_x, offset_y : float
        Manual X/Y offsets applied to 920 nm centroids.
    method : str
        Matching algorithm ('Hungarian' or 'Nearest').

    Returns
    -------
    dict
        Mapping of 800→920 track IDs for the given slices.
    """
    
    c8 = compute_centroids_from_masks("800", z800, VIEW_DIR, S2_8, S2_9)
    c9 = compute_centroids_from_masks("920", z920, VIEW_DIR, S2_8, S2_9)

    if c8.empty or c9.empty:
        print(f"⚠️ No centroids at Z800={z800} or Z920={z920}")
        return {}

    c9 = c9.copy()
    c9["x_ref"] += offset_x
    c9["y_ref"] += offset_y

    if method == "Nearest":
        mapping = _nearest_neighbor_match(c8, c9, max_dist=max_dist)
    else:
        mapping = _hungarian_match(c8, c9, max_dist=max_dist)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs[0].scatter(c8["x_ref"], c8["y_ref"], c="blue", label=f"800 Z{z800}", alpha=0.6)
    axs[0].scatter(c9["x_ref"], c9["y_ref"], c="green", label=f"920 Z{z920}", alpha=0.6)

    for t800, t920 in mapping.items():
        p8 = c8[c8.track_id == t800][["x_ref", "y_ref"]].values[0]
        p9 = c9[c9.track_id == t920][["x_ref", "y_ref"]].values[0]
        axs[0].plot([p8[0], p9[0]], [p8[1], p9[1]], "k--", alpha=0.5)
        axs[0].text(p8[0], p8[1] -5 , str(t800), color="blue", fontsize=10)
        axs[0].text(p9[0], p9[1] + 15, str(t920), color="green", fontsize=10)

    axs[0].invert_yaxis()
    axs[0].set_title(f"Centroids (800 Z{z800} vs 920 Z{z920})\nOffset X={offset_x}, Y={offset_y}, Method={method}")
    axs[0].legend()

    if z800 in raw800_map:
        axs[1].imshow(io.imread(raw800_map[z800]), cmap="gray")
        axs[1].set_title(f"800 nm Z={z800}")
    if z920 in raw920_map:
        axs[2].imshow(io.imread(raw920_map[z920]), cmap="gray")
        axs[2].set_title(f"920 nm Z={z920}")

    plt.show()

    if mapping:
        line = ", ".join([f"{k}:{mapping[k]}" for k in sorted(mapping.keys())])
        print(f"✅ Found {len(mapping)} matches ({method}) at max_dist={max_dist}: {line}")
    else:
        print(f"✅ Found 0 matches at max_dist={max_dist}")
    return mapping



# ---------- Auto build mapping ----------

def auto_build_mapping(cent8, cent9, max_dist=30,
                       offset_x=0.0, offset_y=0.0,
                       method="Hungarian"):
    """
    Automatically compute centroid correspondences between two slices.

    Given two centroid DataFrames (from 800 and 920 nm), this function
    shifts the 920 positions by a defined offset and finds optimal
    matches between the two datasets using the selected algorithm.

    Returns
    -------
    dict[int, int]
        Mapping of track_800 → track_920.
    """
    if cent8.empty or cent9.empty:
        return {}
    cent9 = cent9.copy()
    cent9["x_ref"] += offset_x
    cent9["y_ref"] += offset_y
    if method == "Nearest":
        return _nearest_neighbor_match(cent8, cent9, max_dist)
    else:
        return _hungarian_match(cent8, cent9, max_dist)



# ---------- Consensus mapping across slices ----------

def consensus_mapping_auto(cent8_all, cent9_all, z800_list, z920_list,
                           max_dist=30, offset_x=0.0, offset_y=0.0,
                           method="Hungarian"):
    """
    Build a global and consistent 800→920 track mapping across multiple Z-slices.

    This method compares centroids across nearby slices (Z-1, Z, Z+1),
    counts how often each 800→920 pair is matched, and then selects
    the most consistent correspondences by majority voting.

    Returns
    -------
    tuple
        per800 : Counter dictionary with all match votes per 800 track.
        mapping_init : dict with the initial consensus 800→920 mapping.
    """
    votes = Counter()
    for z800 in z800_list:
        for dz in [-1, 0, +1]:
            z920 = z800 + dz
            if z920 not in z920_list:
                continue
            c8 = cent8_all[cent8_all["z"] == z800]
            c9 = cent9_all[cent9_all["z"] == z920]
            if c8.empty or c9.empty:
                continue
            mapping = auto_build_mapping(c8, c9, max_dist, offset_x, offset_y, method)
            for t800, t920 in mapping.items():
                votes[(t800, t920)] += 1

    per800 = defaultdict(Counter)
    per920 = defaultdict(Counter)
    for (t800, t920), count in votes.items():
        per800[t800][t920] += count
        per920[t920][t800] += count
        

    best_per800 = {}
    for t800, cnts in per800.items():
        if not cnts:
            continue
        t920_best, v_best = cnts.most_common(1)[0]
        best_per800[t800] = (t920_best, v_best)


    winners_for_920 = {}
    for t800, (t920, v) in best_per800.items():
        if t920 not in winners_for_920:
            winners_for_920[t920] = (t800, v)
        else:
            cur_t800, cur_v = winners_for_920[t920]
            if v > cur_v:
                winners_for_920[t920] = (t800, v)


    mapping_init = {t800: t920 for t920, (t800, v) in winners_for_920.items()}

    return per800, mapping_init
