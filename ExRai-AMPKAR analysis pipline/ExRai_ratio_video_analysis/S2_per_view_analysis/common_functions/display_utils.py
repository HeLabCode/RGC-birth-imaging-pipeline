import numpy as np
from skimage import io
import os
import pandas as pd
import cv2
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import measure
import ipywidgets as widgets
from IPython.display import display, clear_output
from S6_size_ratio_subtraction.size_utils import get_best_slice_from_maskstack


# -------- Labeled region overlay visualization -------- 

def colored_overlay_from_labels(labels, base_8u):
    """
    Create a color overlay of labeled regions on a grayscale base image.

    Each region in `labels` is assigned a random RGB color, and its label
    number is rendered at the region centroid. The overlay is blended
    with the grayscale base image for visual clarity.

    Parameters
    ----------
    labels : ndarray (int)
        2D label map where 0 = background and positive integers represent regions.
    base_8u : ndarray (uint8)
        Grayscale base image (8-bit) to overlay on top of.

    Returns
    -------
    overlay : ndarray (uint8)
        RGB image showing colored regions with numeric labels.
    """
    h, w = labels.shape
    rng = np.random.default_rng(42)
    colors = np.zeros((labels.max() + 1, 3), dtype=np.uint8)

    for i in range(1, labels.max() + 1):
        colors[i] = rng.integers(50, 255, size=3, dtype=np.uint8)

    overlay = cv2.cvtColor(base_8u, cv2.COLOR_GRAY2BGR)
    overlay = (overlay * 0.4).astype(np.uint8)
    for i in range(1, labels.max() + 1):
        overlay[labels == i] = (0.9 * overlay[labels == i] + 0.4 * colors[i]).astype(np.uint8)

    for p in measure.regionprops(labels):
        r, c = map(int, p.centroid)
        cv2.putText(overlay, str(p.label), (c, r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(overlay, str(p.label), (c, r),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return overlay


# -------- Random Color generator -------- 

def random_colors(n, seed=42):
    """
    Generate a list of bright random RGB colors for visualization.

    Parameters
    ----------
    n : int
        Number of colors to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of tuple(int, int, int)
        List of RGB color tuples.
    """
    rng = np.random.default_rng(seed)
    return [tuple(map(int, rng.integers(60, 255, size=3))) for _ in range(max(1, n))]



# -------- Overlay color --------  

def overlay_slice(base8, labels, rows_z, colors):
    """
    Overlay colored tracked regions on a grayscale image.

    Each region (label) is colored based on its track ID and annotated
    with the track number for visualization.

    Parameters
    ----------
    base8 : ndarray
        Grayscale 8-bit image.
    labels : ndarray
        2D label map (0 = background).
    rows_z : DataFrame
        Tracks for this Z-slice (must include 'particle', 'label_id', 'x', 'y').
    colors : list of tuple
        List of RGB colors, one per track.

    Returns
    -------
    ndarray
        RGB overlay image with colors and track IDs.
    """
    S2_FONT_SCALE = 0.5
    ov = cv2.cvtColor((base8 * 0.35).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    lab2col = {int(r.label_id): colors[int(r.particle) % len(colors)] for _, r in rows_z.iterrows()}
    for lid, col in lab2col.items():
        ov[labels == lid] = (ov[labels == lid] * 0.3 + np.array(col, np.uint8)).astype(np.uint8)
    for _, r in rows_z.iterrows():
        x, y = float(r.x), float(r.y)
        pid = int(r.particle)
        cv2.putText(ov, str(pid), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, S2_FONT_SCALE  , (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ov, str(pid), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,  S2_FONT_SCALE , (0, 0, 0), 1, cv2.LINE_AA)
    return ov



# ---------- Contour finder ----------

def find_contours_safe(mask):
    """
    Wrapper around OpenCV’s `findContours` that ensures consistent output format.

    Handles differences between OpenCV versions that may return 2 or 3 values.
    Returns both contour coordinates and hierarchy safely.

    Parameters
    ----------
    mask : ndarray (2D)
        Binary image mask (uint8 or bool).

    Returns
    -------
    contours : list of ndarray
        Detected contours (each an Nx1x2 array of (x,y) coordinates).
    hierarchy : ndarray
        Contour hierarchy array from OpenCV.
    """
    result = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 2:
        contours, hierarchy = result
    elif len(result) == 3:
        _, contours, hierarchy = result
    else:
        raise RuntimeError("Unexpected number of return values from cv2.findContours")
    return contours, hierarchy



# ---------- Color composite overlay ----------

def build_composite_best(res, mask_map_800, mask_map_920, max_ratio=5.0, show_920=True):
    """
    Build a color composite overlay visualizing cross-channel matches.

    Each 800 nm cell is drawn in blue. Its matched 920 nm cell is drawn
    in green (accepted) or red (rejected) depending on the area/intensity
    ratio. Contours and IDs are overlaid for visual verification.

    Parameters
    ----------
    res : pandas.DataFrame
        Table containing columns ['track_800', 'track_920', 'ratio_symmetric'].
    mask_map_800, mask_map_920 : dict
        Dictionaries mapping track IDs → mask stack file paths for each channel.
    max_ratio : float, optional
        Maximum allowed ratio between 800/920 areas; above this, pairs are rejected.
    show_920 : bool, optional
        If False, only 800 nm cells are drawn.

    Returns
    -------
    canvas : ndarray (H, W, 3)
        RGB composite image.
    accepted : list of tuples
        [(track_800, track_920, ratio)] for accepted pairs.
    rejected : list of tuples
        [(track_800, track_920, ratio)] for rejected pairs.
    """
    sample_path = next(iter(mask_map_800.values()))
    sample = io.imread(sample_path)
    if sample.ndim == 2:
        H, W = sample.shape
    else:
        H, W = sample.shape[1:]
    canvas = np.zeros((H, W, 3), np.uint8)

    accepted, rejected = [], []

    for r in res.itertuples(index=False):
        t800, t920, ratio = r.track_800, r.track_920, r.ratio_symmetric
        mask8, _, _ = get_best_slice_from_maskstack(mask_map_800[t800])
        mask9, _, _ = get_best_slice_from_maskstack(mask_map_920[t920])

        if mask8 is None or mask9 is None:
            continue

        canvas[mask8.astype(bool), 2] = 255

        if show_920:
            if ratio <= max_ratio:

                canvas[mask9.astype(bool)] = (0,255,0)
                accepted.append((t800, t920, ratio))

                contours, _ = find_contours_safe(mask9)
                cv2.drawContours(canvas, contours, -1, (0,0,0), 1)

                M = cv2.moments(mask8)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    cv2.putText(canvas, str(t800), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255,255,255), 1, cv2.LINE_AA)

            else:
                canvas[mask9.astype(bool)] = (255,0,0)
                rejected.append((t800, t920, ratio))

                contours, _ = find_contours_safe(mask9)
                cv2.drawContours(canvas, contours, -1, (0,0,0), 1)

                M8 = cv2.moments(mask8)
                if M8["m00"] > 0:
                    cx, cy = int(M8["m10"]/M8["m00"]), int(M8["m01"]/M8["m00"])
                    cv2.putText(canvas, str(t800), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255,255,255), 1, cv2.LINE_AA)

                M9 = cv2.moments(mask9)
                if M9["m00"] > 0:
                    cx, cy = int(M9["m10"]/M9["m00"]), int(M9["m01"]/M9["m00"])
                    cv2.putText(canvas, str(t920), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0,0,255), 1, cv2.LINE_AA)

        else:
            contours, _ = find_contours_safe(mask8)
            cv2.drawContours(canvas, contours, -1, (0,0,0), 1)
            M = cv2.moments(mask8)
            if M["m00"] > 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.putText(canvas, str(t800), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255,255,255), 1, cv2.LINE_AA)

    return canvas, accepted, rejected



# ----------  Interactive widget for 800↔920 cell pairs ----------

def interactive_area_ratio_viewer(res, mask_map_800, mask_map_920, save_path=None):
    """
    Interactive widget for filtering and visualizing 800↔920 cell pairs.

    Allows dynamic adjustment of the acceptance threshold (ratio) and
    optional display of the 920 nm channel. Accepted/rejected pairs are
    visualized with color overlays, and accepted pairs can be saved to CSV.

    Parameters
    ----------
    res : pandas.DataFrame
        Table with columns ['track_800', 'track_920', 'ratio_symmetric'].
    mask_map_800, mask_map_920 : dict
        Mapping of track IDs → mask stack file paths.
    save_path : str, optional
        Output path to save accepted pairs (CSV).

    Outputs
    -------
    - Interactive Jupyter widget with:
        • Ratio slider
        • Toggle for 920 visibility
        • Save button
    - Saved CSV (if `save_path` is given)
    """
    ratio_slider = widgets.FloatSlider(value=5.0, min=1.0, max=20.0, step=0.5,
                                       description="Max Ratio", readout_format=".1f")
    show_920_toggle = widgets.Checkbox(value=True, description="Show 920 channel")
    save_btn = widgets.Button(description="💾 Save Filtered Map", button_style="success")
    out = widgets.Output()

    accepted_global = []

    def update(change=None):
        nonlocal accepted_global
        max_ratio = ratio_slider.value
        show_920 = show_920_toggle.value
        canvas, accepted, rejected = build_composite_best(res, mask_map_800, mask_map_920,
                                                          max_ratio=max_ratio, show_920=show_920)
        accepted_global = accepted

        with out:
            clear_output(wait=True)
            plt.figure(figsize=(8,8))
            plt.imshow(canvas)
            plt.axis("off")
            plt.title(f"Accepted: {len(accepted)} | Rejected: {len(rejected)} (threshold={max_ratio:.1f})")
            plt.show()

    def on_save(_):
        if not accepted_global:
            print("⚠️ No accepted pairs to save!")
            return
        df_out = pd.DataFrame([{"track_800": t800, "track_920": t920, "ratio": ratio}
                               for t800, t920, ratio in accepted_global])
        if save_path:
            df_out.to_csv(save_path, index=False)
            print(f"💾 Saved {len(df_out)} filtered pairs to {save_path}")
        else:
            print(df_out.head())

    ratio_slider.observe(update, names="value")
    show_920_toggle.observe(update, names="value")
    save_btn.on_click(on_save)

    display(widgets.VBox([ratio_slider, show_920_toggle, save_btn, out]))
    update()



# ----------  Safe print ----------

def print_safe(msg):
    """
    Safe print wrapper for logging messages in all pipeline stages.

    Purpose:
        Ensures messages are printed even if the pipeline logger 
        (used in interactive or notebook contexts) is unavailable.

    Input:
        msg : str
            Message to print or log.

    Behavior:
        - Tries to use the active pipeline's logger (if present).
        - Falls back to standard print() if the logger is not available.
    """
    try:
        print(msg)
    except Exception:
        print(msg)
        
        
        
# ----------  Mapping scalar values to RGB colors ----------

def _make_colormap_sampler(name="viridis", n=1024):
    """
    Create a fast color sampler for mapping scalar values to RGB colors.

    Purpose
    -------
    Builds a small, precomputed lookup table (LUT) from a matplotlib colormap,
    allowing for rapid conversion of normalized scalar values (0→1) into
    RGB color triplets without repeatedly calling matplotlib functions.

    Parameters
    ----------
    name : str, optional
        Name of the matplotlib colormap to sample from (default: 'viridis').
        Common choices: 'jet', 'magma', 'plasma', 'coolwarm', etc.
    n : int, optional
        Number of samples to precompute for the LUT (default: 1024).

    Returns
    -------
    sampler : function
        A callable that maps a NumPy array or scalar `v` ∈ [0,1] → RGB uint8 array
        of shape (len(v), 3). Values outside [0,1] are automatically clipped.
    """
    cmap = cm.get_cmap(name, n)
    lut = (cmap(np.linspace(0,1,n))[:, :3] * 255.0).astype(np.uint8)  
    def sampler(v):
        v = np.clip(v, 0.0, 1.0)
        idx = np.minimum((v*(n-1)).astype(int), n-1)
        return lut[idx]
    return sampler




def _compute_unions_for_tracks(z_order, size_hw, track_ids_800, z_min, z_max, mask_map_800):
    """
    Compute 2D union masks for each tracked object over a given Z-range.

    Purpose
    -------
    For each 800 nm track, this function loads its per-slice binary mask stack,
    merges (logical OR) all masks between z_min and z_max, and records both
    the combined union mask and its total area. Used later for projection
    rendering and ratio visualization.

    Parameters
    ----------
    z_order : list[int]
        Ordered list of Z-slice indices to iterate over.
    size_hw : tuple[int, int]
        Image dimensions (height, width) for preallocating union arrays.
    track_ids_800 : list[int]
        List of 800 nm track IDs to process.
    z_min, z_max : int
        Inclusive bounds of the Z-range used for union accumulation.
    mask_map_800 : dict[int, str]
        Dictionary mapping track IDs to corresponding maskstack file paths.

    Returns
    -------
    list[tuple[int, np.ndarray, int]]
        For each track ID, returns a tuple:
            (track_id, union_mask_bool[H,W], union_area_int)
    """
    H, W = size_hw
    H, W = int(H), int(W)
    result = []
    for tid in track_ids_800:
        p = mask_map_800.get(int(tid))
        if not p or not os.path.exists(p):
            print_safe(f"WARNING: missing mask stack for 800 track {tid}: {p or '(not indexed)'}")
            result.append((int(tid), np.zeros((H,W), bool), 0))
            continue
        try:
            mstk = io.imread(p)
        except Exception as e:
            print_safe(f"WARNING: failed to read mask stack {p}: {e}")
            result.append((int(tid), np.zeros((H,W), bool), 0))
            continue
        if mstk.ndim == 2:
            mstk = mstk[np.newaxis, ...]
        union = np.zeros((H, W), bool)

        for zi, z in enumerate(z_order):
            if zi >= mstk.shape[0]:
                break
            if z_min <= z <= z_max:
                union |= (mstk[zi] > 0)
        area = int(union.sum())
        result.append((int(tid), union, area))
    return result
