# =========================================================
#  S7 — Manual link, ratio computation & projection overlay
# =========================================================
import os, cv2, pickle
import numpy as np
import pandas as pd
from skimage import io
from matplotlib.colors import LinearSegmentedColormap

from common_functions.display_utils import print_safe
from common_functions.geometry_utils import _estimate_perZ_transforms_from_json
from common_functions.data_utils import (
    _load_intensity_table,
    load_manual_map,
    index_maskstacks,
    load_raw_map,
    _per_track_centroids_perZ_ref,
)
from common_functions.display_utils import _compute_unions_for_tracks


def run_S7_manual_link_ratio_and_projection(
    VIEW_DIR, S2_8, S2_9, S5_8, S5_9, S6, S7,
    pickle_path=None,
    RATIO_VMIN=0.2,
    RATIO_VMAX=0.4,
):
    """
    Execute the S7 pipeline: manual track linking, ratio computation, and pseudo-color projection.
    """



    MAPPING_CSV = os.path.join(S6, "area_filtered_mapping.csv")  

    Z_MIN, Z_MAX = 1, 18

    RATIO_USE_Z_RANGE = False
    pkl_file_name = "view1.pkl"

    INT_SINGLE_FILE_800 = os.path.join(S5_8, "800_S5_means_bgsub.csv")
    INT_SINGLE_FILE_920 = os.path.join(S5_9, "920_S5_means_bgsub.csv")
    INT_DIR_800 = S7  
    INT_DIR_920 = S7

    # --- Visualization ---
    CMAP_NAME = "grad" 
    DRAW_OUTLINES = True
    SAVE_PER_CELL_MASKS = True
    PER_CELL_MASK_DIR = os.path.join(S7, "S7_proj_masks")

    FORCE_DRAW_CENTROID_IF_NO_MASK = True
    FALLBACK_DISK_RADIUS = 6  # px

    ALPHA_BLEND = False
    ALPHA_PER_CELL = 0.6  

    os.makedirs(S7, exist_ok=True)

    raw800_map, z800_list, size_hw = load_raw_map("800", VIEW_DIR)
    raw920_map, z920_list, _ = load_raw_map("920", VIEW_DIR)
    z_common = sorted(set(z800_list).intersection(z920_list))
    if not z_common:
        raise RuntimeError("No common Z numbers between 800 and 920.")
    z_min = max(Z_MIN, z_common[0]); z_max = min(Z_MAX, z_common[-1])
    print_safe(f"Using Z range: [{z_min}, {z_max}] (within common Zs {z_common[0]}..{z_common[-1]})")

    A800_by_z, _ = _estimate_perZ_transforms_from_json("800", z800_list, S2_8, S2_9)
    mask_map_800 = index_maskstacks("800", S2_8, S2_9)
    cent8_z = _per_track_centroids_perZ_ref(
        S2_8, S2_9, "800", z800_list, A800_by_z, mask_map=mask_map_800
    )


    m = load_manual_map(MAPPING_CSV)  
    track_ids_800 = list(m.keys())



    tab800 = _load_intensity_table(
        channel="800",
        INT_SINGLE_FILE_800=INT_SINGLE_FILE_800,
        INT_SINGLE_FILE_920=INT_SINGLE_FILE_920,
        INT_DIR_800=INT_DIR_800,
        INT_DIR_920=INT_DIR_920,
    )  
    tab920 = _load_intensity_table(
        channel="920",
        INT_SINGLE_FILE_800=INT_SINGLE_FILE_800,
        INT_SINGLE_FILE_920=INT_SINGLE_FILE_920,
        INT_DIR_800=INT_DIR_800,
        INT_DIR_920=INT_DIR_920,
    )   

    if RATIO_USE_Z_RANGE:
        tab800 = tab800[(tab800["z"] >= z_min) & (tab800["z"] <= z_max)].copy()
        tab920 = tab920[(tab920["z"] >= z_min) & (tab920["z"] <= z_max)].copy()


    max800 = tab800.groupby("track_id", as_index=False)["value"].max().rename(columns={"value": "max800"})
    max920 = tab920.groupby("track_id", as_index=False)["value"].max().rename(columns={"value": "max920"})

    rows = []
    for t800, t920 in m.items():
        v800 = max800[max800["track_id"] == t800]["max800"]
        v920 = max920[max920["track_id"] == t920]["max920"]
        mv800 = float(v800.iloc[0]) if len(v800) else np.nan
        mv920 = float(v920.iloc[0]) if len(v920) else np.nan
        ratio = mv920 / mv800 if (mv800 is not None and mv800 > 0 and np.isfinite(mv800) and np.isfinite(mv920)) else np.nan

        sub = cent8_z[
            (cent8_z["track_id"] == t800) &
            (cent8_z["z"] >= z_min) &
            (cent8_z["z"] <= z_max)
        ]
        cx = float(sub["x_ref"].mean()) if len(sub) else np.nan
        cy = float(sub["y_ref"].mean()) if len(sub) else np.nan

        rows.append({
            "track_800": int(t800),
            "track_920": int(t920),
            "z_min": int(z_min),
            "z_max": int(z_max),
            "max800": mv800,
            "max920": mv920,
            "ratio_920_over_800": ratio,
            "centroid_x": cx,
            "centroid_y": cy,
        })
    res = pd.DataFrame(rows).sort_values("track_800").reset_index(drop=True)

    # ---------- unions & projection ----------
    unions = _compute_unions_for_tracks(z800_list, size_hw, track_ids_800, z_min, z_max, mask_map_800)

    n_nonzero = sum(1 for _, _, a in unions if a > 0)

    cmap = LinearSegmentedColormap.from_list(
        "grad",
        ["#006400", "#66A05B", "#FFA500", "#FF8C00"],
    )

    def sampler(vals):
        """
        vals: array of normalized values in [0,1]
        returns: uint8 RGB array in [0,255], shape (..., 3)
        """
        vals = np.asarray(vals, dtype=np.float32)
        vals = np.clip(vals, 0.0, 1.0)
        rgba = cmap(vals)                
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        return rgb

    H, W = int(size_hw[0]), int(size_hw[1])
    vis = np.zeros((H, W, 3), np.uint8)  
    ratio_map = {int(r.track_800): float(r.ratio_920_over_800) for r in res.itertuples(index=False)}

    def _norm(v):
        if not np.isfinite(v):
            return np.nan
        return (v - RATIO_VMIN) / (RATIO_VMAX - RATIO_VMIN)

    def _ratio_to_bgr(r):
        if not np.isfinite(r):
            return np.array([30, 30, 30], np.uint8)
        nv = _norm(r)
        if not np.isfinite(nv):
            return np.array([30, 30, 30], np.uint8)
        rgb = sampler(np.array([nv]))[0]
        return rgb[::-1] 

    if ALPHA_BLEND:
        canvas = vis.astype(np.float32)
        for tid, union, area in unions:
            if area == 0:
                continue
            bgr = _ratio_to_bgr(ratio_map.get(tid, np.nan)).astype(np.float32)
            mask = union.astype(np.float32)[..., None]
            canvas = canvas * (1.0 - mask * ALPHA_PER_CELL) + bgr * (mask * ALPHA_PER_CELL)
        vis = np.clip(canvas, 0, 255).astype(np.uint8)
    else:
        unions_sorted = sorted(unions, key=lambda x: x[2], reverse=True)
        for tid, union, _area in unions_sorted:
            if _area == 0:
                continue
            bgr = _ratio_to_bgr(ratio_map.get(int(tid), np.nan))
            vis[union.astype(bool)] = bgr

    if DRAW_OUTLINES:
        all_union = np.zeros((H, W), np.uint8)
        for _, u, a in unions:
            if a > 0:
                all_union |= (u.astype(np.uint8))
        if all_union.any():
            kernel = np.ones((3, 3), np.uint8)
            borders = cv2.dilate(all_union, kernel, 1) - all_union
            vis[borders > 0] = (0, 0, 0)

    if FORCE_DRAW_CENTROID_IF_NO_MASK:
        empty_tids = [tid for tid, _, area in unions if area == 0]
 
        for tid in empty_tids:
            row = res[res["track_800"] == tid]
            if len(row) == 0:
                continue
            cx, cy = float(row.iloc[0]["centroid_x"]), float(row.iloc[0]["centroid_y"])
            if np.isfinite(cx) and np.isfinite(cy):
                bgr = _ratio_to_bgr(ratio_map.get(tid, np.nan)).tolist()
                cv2.circle(vis, (int(cx), int(cy)), FALLBACK_DISK_RADIUS, bgr, thickness=-1, lineType=cv2.LINE_AA)

    # --- save a copy before drawing numeric labels ---
    vis_no_labels = vis.copy()
    out_png_nolabels = os.path.join(S7, "S7_ratio_projection_nolabels.tif")
    cv2.imwrite(out_png_nolabels, vis_no_labels)

    for r in res.itertuples(index=False):
        if not np.isfinite(r.centroid_x) or not np.isfinite(r.centroid_y):
            continue
        x, y = int(r.centroid_x), int(r.centroid_y)
        txt = f"{r.track_800}"
        cv2.putText(vis, txt, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, txt, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_csv = os.path.join(S7, "S7_manual_ratio_results.csv")
    res.to_csv(out_csv, index=False)

    out_png = os.path.join(S7, "S7_ratio_projection.tif")
    cv2.imwrite(out_png, vis)


    # --- separate vertical colorbar image ---

    bar_w, bar_h = 20, 360
    pad = 8
    tick_w = 6
    label_w = 40

    canvas_h = bar_h + 2 * pad
    canvas_w = bar_w + 2 * pad + tick_w + label_w
    cbar = np.full((canvas_h, canvas_w, 3), 255, np.uint8)

    y_top = pad
    y_bot = pad + bar_h - 1

    for j in range(bar_h):

        frac = j / (bar_h - 1)
        v = RATIO_VMIN + frac * (RATIO_VMAX - RATIO_VMIN)
        norm_val = _norm(v)
        rgb = sampler(np.array([norm_val]))[0]  
        y = y_bot - j
        cbar[y:y+1, pad:pad + bar_w, :] = rgb[::-1] 

    x_right = pad + bar_w
    tick_vals = [
        RATIO_VMIN,
        0.5 * (RATIO_VMIN + RATIO_VMAX),
        RATIO_VMAX,
    ]

    for v in tick_vals:
        frac = _norm(v)
        y = y_bot - int(frac * (bar_h - 1))
        y = int(np.clip(y, y_top, y_bot))
        label = f"{v:.2f}"
        cv2.line(cbar, (x_right, y), (x_right + tick_w, y), (0, 0, 0), 1)
        cv2.putText(
            cbar, label,
            (x_right + tick_w + 4, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 0, 0), 1, cv2.LINE_AA
        )

    out_cbar = os.path.join(S7, "S7_colorbar.tif")
    ok = cv2.imwrite(out_cbar, cbar)


    # --- PKL: save payload used to render S7 projection ---
    if pickle_path is not None:
        H, W = vis.shape[:2]

        ratio_map_local = {int(r.track_800): float(r.ratio_920_over_800) for r in res.itertuples(index=False)}
        centroid_map_local = {
            int(r.track_800): (float(r.centroid_x), float(r.centroid_y))
            for r in res.itertuples(index=False)
        }

        payload = {
            "z_min": int(z_min),
            "z_max": int(z_max),
            "size_hw": (int(H), int(W)),
            "cmap": CMAP_NAME,
            "vmin": float(RATIO_VMIN),
            "vmax": float(RATIO_VMAX),
            "alpha_blend": bool(ALPHA_BLEND),
            "alpha_per_cell": float(ALPHA_PER_CELL),
            "draw_outlines": bool(DRAW_OUTLINES),
            "force_draw_centroid_if_no_mask": bool(FORCE_DRAW_CENTROID_IF_NO_MASK),
            "fallback_disk_radius": int(FALLBACK_DISK_RADIUS),
            "label_image": None,   
            "cells": [
                {
                    "cell_id": int(tid),
                    "centroid": tuple(centroid_map_local.get(int(tid), (np.nan, np.nan))),
                    "area": int(area),
                    "ratio": float(ratio_map_local.get(int(tid), np.nan)),
                    "union": union,
                }
                for (tid, union, area) in unions
            ],
        }

        if not ALPHA_BLEND:
            payload["paint_order_by_area"] = [
                int(tid) for (tid, _u, _a) in sorted(unions, key=lambda x: x[2], reverse=True)
            ]

        with open(pickle_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print_safe(f"S7: saved project")
