# ============================
# S2: Manual Anchor & Tracking
# ============================

import os
import json
import cv2
import numpy as np
import pandas as pd
from skimage import io, measure
import trackpy as tp
from common_functions.data_utils import load_s1_items, normalize_to_8u, export_track_masks_and_index
from common_functions.display_utils import random_colors, overlay_slice
from S2_tracking.tracking_tools import collect_three_anchors, estimate_transforms

# ============================
# MAIN FUNCTION
# ============================


def run_S2_manual_anchor_3pts(channel, VIEW_DIR, S1_8, S1_9, S2_8, S2_9):
    """
    Run the full S2 tracking pipeline for the selected channel.

    This function loads segmented regions from S1, lets the user define
    3-point anchors for Z alignment, estimates per-slice transformations,
    tracks objects through Z using TrackPy, and exports:
      - Per-slice overlays
      - Linked tracking CSVs
      - Per-track binary masks and index CSVs

    Parameters
    ----------
    channel : str
        Channel name ("800" or "920").
    """

    # -------- Parameters --------
    S2_SEARCH_RANGE = 20       # px between adjacent Z slices
    S2_MEMORY       = 2        # missing Z planes allowed for same track
    S2_MIN_AREA     = 10       # drop tiny objects
    S2_SAVE_FORMAT  = "tif"    # overlay format  
    S2_SEED         = 42


    for d in [S2_8, S2_9]:
        os.makedirs(d, exist_ok=True)

    # --- Main ---
    print(f"S2[{channel}]: load S1…")
    items = load_s1_items(channel, S1_8, S1_9, VIEW_DIR)
    if not items:
        print("⚠️ No items found — skipping.")
        return

    frames8 = [(z, normalize_to_8u(raw)) for z, _, raw in items]
    z_order = [z for z, _, _ in items]
    out_dir = S2_8 if channel == "800" else S2_9

    anchors_json = os.path.join(out_dir, f"{channel}_S2_anchors_3pt.json")

    if os.path.exists(anchors_json):
        with open(anchors_json, "r") as f:
            data = json.load(f)

        anchors = {
            int(k): [tuple(p) for p in v]
            for k, v in data.items()
        }
        print(f"  loaded anchors for {len(anchors)} slices")

    else:
        print("👉 Click THREE anchors (A,B,C) per Z slice")
        anchors = collect_three_anchors(frames8, scale=1.0)

        with open(anchors_json, "w") as f:
            json.dump(
                {
                    int(k): [[float(x), float(y)] for (x, y) in v]
                    for k, v in anchors.items()
                },
                f,
                indent=2,
            )
        
    A_by_z = estimate_transforms(z_order, anchors, ref_z=z_order[0])
    rows = []
    for z, labels, raw in items:
        A = A_by_z[z]
        for p in measure.regionprops(labels):
            if p.area < S2_MIN_AREA:
                continue
            x0, y0 = float(p.centroid[1]), float(p.centroid[0])
            x1 = A[0, 0] * x0 + A[0, 1] * y0 + A[0, 2]
            y1 = A[1, 0] * x0 + A[1, 1] * y0 + A[1, 2]
            rows.append({
                "frame": int(z), "z": int(z), "x": x1, "y": y1,
                "label_id": int(p.label), "area": int(p.area)
            })
    feat = pd.DataFrame(rows)
    if feat.empty:
        print("⚠️ No features to track — skipping.")
        return
    
    linked = tp.link_df(
        feat,
        search_range=S2_SEARCH_RANGE,
        memory=S2_MEMORY,
        adaptive_step=0.95,
        neighbor_strategy="KDTree"
    )
    linked.to_csv(os.path.join(out_dir, f"{channel}_S2_tracks_linked.csv"), index=False)
    colors = random_colors(linked.particle.nunique(), seed=S2_SEED)
    by_z = {z: linked[linked.z == z] for z in z_order}
    for z, labels, raw in items:
        base8 = normalize_to_8u(raw)
        rows_z = by_z.get(z, pd.DataFrame(columns=["particle", "label_id"]))
        ov = overlay_slice(base8, labels, rows_z, colors)
        out = os.path.join(out_dir, f"{channel}_tracks_Z{z:03d}.{S2_SAVE_FORMAT}")
        if S2_SAVE_FORMAT == "png":
            cv2.imwrite(out, ov)
        else:
            io.imsave(out, cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
    n = export_track_masks_and_index(channel, items, linked, out_dir)
    print(f"✅ Exported {n} tracks (mask stacks + per-Z masks + index CSV).")
