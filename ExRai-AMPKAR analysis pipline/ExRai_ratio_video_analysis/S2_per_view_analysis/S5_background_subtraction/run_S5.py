# ===============================
# S5: Local background subtraction (per-cell, per-slice)
# ===============================

import os, glob
import numpy as np
import pandas as pd
from skimage import io
from common_functions.data_utils import load_raw_map, track_id_from_filename

print("\n🔆 S5: Computing local background per cell (per-slice)...\n")

# ============================
# MAIN FUNCTION
# ============================

def run_S5_background_subtraction(channel, VIEW_DIR, S2_8, S2_9, S5_8, S5_9):
    """
    Compute per-cell mean intensities and subtract a local background.

    Method
    ------
    • For each Z-slice, load the raw image.
    • For each cell, compute the mean raw intensity inside the mask.
    • Define a local background annulus centered at the cell centroid:
        - Outer radius = RING_RADIUS (pixels).
        - Inner radius = (max cell distance from centroid) + INNER_GAP_PX.
          This excludes the cell itself and a ~5 px halo.
    • Exclude pixels belonging to any other cell from this annulus
      using a per-slice union mask.
    • Background for that cell is the mean raw intensity over the remaining
      annulus pixels.
    • If the annulus is empty (crowded region or near borders),
      fall back to the slice-wide mean raw intensity.

    Outputs
    -------
    • {channel}_S5_means_raw.csv   : per-slice, per-cell raw means
    • {channel}_S5_means_bgsub.csv : per-slice, per-cell (raw - background)
    """

    s2_dir = S2_8 if channel == "800" else S2_9
    s5_dir = S5_8 if channel == "800" else S5_9
    os.makedirs(s5_dir, exist_ok=True)

    # ---- Load raw stack ----
    raw_map, zlist, (H, W) = load_raw_map(channel, VIEW_DIR)
    stack = np.stack([io.imread(raw_map[z]).astype(np.float32) for z in zlist])
    expected_Z = len(zlist)

    # ---- Load per-track masks ----
    stack_paths = sorted(glob.glob(os.path.join(s2_dir, f"{channel}_track*_maskstack.tif")))
    if not stack_paths:
        raise RuntimeError(f"No track mask stacks found in {s2_dir} for channel {channel}")

    track_ids, track_masks = [], []
    for p in stack_paths:
        tid = track_id_from_filename(p)
        mstk = io.imread(p)
        if mstk.ndim == 2:
            mstk = mstk[np.newaxis, ...]
        elif mstk.ndim != 3:
            raise ValueError(f"Unexpected maskstack dims for {p}: {mstk.shape}")
        mstk = (mstk > 0)

        if mstk.shape[0] != expected_Z:
            pad = expected_Z - mstk.shape[0]
            mstk = np.pad(mstk, ((0, max(pad, 0)), (0, 0), (0, 0)), mode="constant")

        track_ids.append(tid)
        track_masks.append(mstk)

    order = np.argsort(track_ids)
    track_ids_sorted = [track_ids[i] for i in order]
    n_tracks = len(track_ids_sorted)

    # ---- Parameters ----
    RING_RADIUS   = 100  # outer radius around centroid in pixels
    INNER_GAP_PX  = 5    # extra halo beyond cell extent (in pixels)

    raw_rows, bgsub_rows = [], []

    # ---- Process each Z slice ----
    for zi, z in enumerate(zlist):
        raw = stack[zi]
        vals_raw = np.full(n_tracks, np.nan)
        vals_bgs = np.full(n_tracks, np.nan)

        slice_mean_bg = float(raw.mean())

        all_cells = np.zeros_like(raw, dtype=bool)
        for idx in range(n_tracks):
            all_cells |= track_masks[order[idx]][zi]

        for idx in range(n_tracks):
            mask = track_masks[order[idx]][zi]
            if not np.any(mask):
                continue

            m_raw = float(raw[mask].mean())

            ys, xs = np.nonzero(mask)
            cy = ys.mean()
            cx = xs.mean()

            dist2_cell = (ys - cy) * (ys - cy) + (xs - cx) * (xs - cx)
            r_cell = float(np.sqrt(dist2_cell.max())) if dist2_cell.size > 0 else 0.0

            r_inner = r_cell + INNER_GAP_PX
            r_outer = RING_RADIUS
            if r_inner >= r_outer:
                r_inner = max(0.0, r_outer - 5.0)

            r_inner2 = r_inner * r_inner
            r_outer2 = r_outer * r_outer

            y0 = max(0, int(cy - r_outer))
            y1 = min(H, int(cy + r_outer) + 1)
            x0 = max(0, int(cx - r_outer))
            x1 = min(W, int(cx + r_outer) + 1)

            if y0 >= y1 or x0 >= x1:
                m_bg = slice_mean_bg
                vals_raw[idx] = m_raw
                vals_bgs[idx] = m_raw - m_bg
                continue

            yy, xx = np.ogrid[y0:y1, x0:x1]
            dy = yy - cy
            dx = xx - cx
            dist2 = dy * dy + dx * dx

            ring_roi = (dist2 <= r_outer2) & (dist2 > r_inner2)

            other_cells = all_cells & ~mask
            other_roi = other_cells[y0:y1, x0:x1]
            ring_roi &= ~other_roi

            if np.any(ring_roi):
                m_bg = float(raw[y0:y1, x0:x1][ring_roi].mean())
            else:
                m_bg = slice_mean_bg

            vals_raw[idx] = m_raw
            vals_bgs[idx] = m_raw - m_bg

        row_r = {"z_number": int(z)}
        row_bgs = {"z_number": int(z)}
        for j, tid in enumerate(track_ids_sorted):
            row_r[f"cell_{tid}"] = vals_raw[j]
            row_bgs[f"cell_{tid}"] = vals_bgs[j]
        raw_rows.append(row_r)
        bgsub_rows.append(row_bgs)

    # ---- Save ----
    df_raw = pd.DataFrame(raw_rows).sort_values("z_number")
    df_bgsub = pd.DataFrame(bgsub_rows).sort_values("z_number")

    df_raw.to_csv(os.path.join(s5_dir, f"{channel}_S5_means_raw.csv"), index=False)
    df_bgsub.to_csv(os.path.join(s5_dir, f"{channel}_S5_means_bgsub.csv"), index=False)

    print(f"S5[{channel}]: processed {len(track_ids_sorted)} cells")
    print(f" → Saved to {s5_dir}\n")
