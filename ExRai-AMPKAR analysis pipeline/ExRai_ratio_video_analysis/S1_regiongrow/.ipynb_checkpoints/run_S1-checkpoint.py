# ============================
# S1: Region Growing Segmentation
# ============================

import os
import cv2
import json
import numpy as np
import pandas as pd
from common_functions.data_utils import load_stack_sorted, normalize_to_8u
from common_functions.display_utils import colored_overlay_from_labels
from S1_regiongrow.segmentation_tools import adaptive_region_grow, watershed_split


# ============================
# MAIN FUNCTION
# ============================

def run_S1_regiongrow(CHANNEL, VIEW_DIR, S1_8, S1_9):
    """
    Run the full S1 region-growing segmentation pipeline for a given channel.

    This function provides an interactive interface for semi-automatic
    segmentation of cells (or regions) across Z-slices. The user clicks
    directly on the image to trigger adaptive region growing, optionally
    splitting connected regions using a two-click watershed approach.

    Features
    --------
    - Interactive region growing from seed points
    - Optional splitting of existing regions via watershed
    - Undo, clear, navigation, and save controls
    - Automatic saving of label maps, overlays, and centroid coordinates

    Controls
    --------
      Left click : grow or split region
      'u'        : undo last region
      'x'        : clear current Z-slice
      's'        : save results (PNG, NPY, JSON, CSV)
      'q' / 'b'  : navigate forward/backward through slices
      'ESC'      : save all and exit

    Parameters
    ----------
    CHANNEL : str
        Channel name (e.g., "800" or "920").

    Outputs
    -------
    - <CHANNEL>_regiongrow_labels_Z###.npy : per-slice label maps
    - <CHANNEL>_regiongrow_overlay_Z###.png : visual overlays
    - <CHANNEL>_regiongrow_centers.json : seed points per Z
    - <CHANNEL>_regiongrow_centers.csv : seed point summary
    """
    
    # -------- Configuration  --------
    stack, zlist, flist = load_stack_sorted(VIEW_DIR, f"{CHANNEL}_ch2_")
    pairs = [(z, None, fpath) for z, fpath in zip(zlist, flist)]

    out_dir = S1_8 if CHANNEL == "800" else S1_9
    os.makedirs(out_dir, exist_ok=True)

    out_json   = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.json")
    out_csv    = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.csv")
    out_overlay = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_overlay_Z{z:03d}.png")
    out_lblnpy  = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_labels_Z{z:03d}.npy")


    # -------- Initialization  --------
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            centers = {int(k): [tuple(p) for p in v] for k, v in json.load(f).items()}
    else:
        centers = {}

    label_maps = {z: np.zeros_like(cv2.imread(rpath, cv2.IMREAD_GRAYSCALE), np.int32)
                for z, _, rpath in pairs}
    active_mask = {z: None for z, _, _ in pairs}
    first_seed = {z: None for z, _, _ in pairs}
    split_contour_visible = {z: None for z, _, _ in pairs}
    
    
    # -------- Interaction  --------
    i = 0
    win = "RegionGrow (2-click): click=add/split | u=undo | x=clear | s=save | q/b nav | ESC quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        nonlocal i
        if event == cv2.EVENT_LBUTTONDOWN:
            z, _, rpath = pairs[i]
            raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
            if raw is None: return
            raw = cv2.GaussianBlur(raw, (3,3), 0)
            centers.setdefault(z, [])

            lbl = label_maps[z]
            mask = active_mask[z]
            split_contour_visible[z] = None

            r, c = int(y), int(x)

            # CASE 1: Add new region
            if mask is None or not mask[r, c]:
                region = adaptive_region_grow(raw, (r, c), rel_drop=0.6, max_radius=25, mask_existing=lbl)
                if np.any(region):
                    new_id = lbl.max() + 1
                    lbl[region.astype(bool)] = new_id
                    active_mask[z] = region.astype(bool)
                    first_seed[z] = (r, c)
                    centers[z].append((r, c))
                    
            # CASE 2: Split existing region
            elif mask[r, c] and first_seed[z] is not None:
                seed1 = first_seed[z]
                seed2 = (r, c)
                ws, contour = watershed_split(mask, seed1, seed2)
                lbl[lbl == lbl[r, c]] = 0
                lbl[ws == 1] = lbl.max() + 1
                lbl[ws == 2] = lbl.max() + 1
                split_contour_visible[z] = contour
                active_mask[z] = None
                first_seed[z] = None
                centers[z].append((r, c))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
                z, _, _ = pairs[i]
                lbl = label_maps[z]
                if lbl.max() > 0:
                    last_id = lbl.max()
                    lbl[lbl == last_id] = 0
                    if z in centers and centers[z]:
                        centers[z].pop()
                active_mask[z] = None
                first_seed[z] = None
                print(f"[Undo] Z{z:03d} last region removed.")


    cv2.setMouseCallback(win, mouse_cb)

    def visualize(raw8, labels):
        return colored_overlay_from_labels(labels, raw8)
    
    # -------- Main  --------
    while True:
        z, _, rpath = pairs[i]
        raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
        raw8 = normalize_to_8u(raw)
        vis = visualize(raw8, label_maps[z])
        header = f"{CHANNEL} Z{z:03d} ({i+1}/{len(pairs)}) | Cells: {label_maps[z].max()}"
        cv2.putText(vis, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        cv2.putText(vis, "click=add/split | u=undo | x=clear | s=save | q/b nav | ESC quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            if i < len(pairs)-1: i += 1
        elif key == ord('b'):
            if i > 0: i -= 1
        elif key == ord('u'):
            z, _, _ = pairs[i]
            lbl = label_maps[z]
            if lbl.max() > 0:
                last_id = lbl.max()
                lbl[lbl == last_id] = 0
                if z in centers and centers[z]:
                    centers[z].pop()
            active_mask[z] = None
            first_seed[z] = None
        elif key == ord('x'):
            z, _, _ = pairs[i]
            centers[z] = []
            label_maps[z][:] = 0
            active_mask[z] = None
            first_seed[z] = None
        elif key == ord('s'):
            z, _, _ = pairs[i]
            labels = label_maps[z]
            overlay = colored_overlay_from_labels(labels, normalize_to_8u(raw))
            cv2.imwrite(out_overlay(z), overlay)
            np.save(out_lblnpy(z), labels)
            with open(out_json, "w") as f:
                json.dump({int(k): v for k,v in centers.items()}, f, indent=2)
            df = pd.DataFrame([(CHANNEL, zz, j, r, c)
                            for zz, plist in centers.items()
                            for j,(r,c) in enumerate(plist)],
                            columns=["channel","Z","point_idx","row","col"])
            df.to_csv(out_csv, index=False)
            print(f"[Saved Z{z:03d}] {len(np.unique(labels))-1} cells -> {out_overlay(z)}")
        elif key == 27:
            with open(out_json, "w") as f:
                json.dump({int(k): v for k,v in centers.items()}, f, indent=2)
            df = pd.DataFrame([(CHANNEL, zz, j, r, c)
                            for zz, plist in centers.items()
                            for j,(r,c) in enumerate(plist)],
                            columns=["channel","Z","point_idx","row","col"])
            df.to_csv(out_csv, index=False)
            print("[Exit] Data saved.")
            break

    cv2.destroyAllWindows()
