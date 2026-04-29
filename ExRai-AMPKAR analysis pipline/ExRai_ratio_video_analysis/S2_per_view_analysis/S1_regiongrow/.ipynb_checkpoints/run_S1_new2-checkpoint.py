# ============================
# S1: Region Growing Segmentation (Hybrid with NN)
# ============================

import os
import cv2
import json
import numpy as np
import pandas as pd

from common_functions.data_utils import load_stack_sorted, normalize_to_8u
from common_functions.display_utils import colored_overlay_from_labels
from S1_regiongrow.segmentation_tools import adaptive_region_grow, watershed_split


# ============================================================
# Helper: choose RR vs NN for a single new region
# ============================================================

def choose_rr_or_nn(rr_mask, nn_labels, seed_rc, nn_min_frac=0.80, below_mode="NN"):

    r0, c0 = seed_rc
    H, W = nn_labels.shape

    if not (0 <= r0 < H and 0 <= c0 < W):
        return rr_mask, "RR (seed out of bounds)", 0.0

    nn_id = nn_labels[r0, c0]
    if nn_id == 0:
        return rr_mask, "RR (no NN at seed)", 0.0

    nn_mask = (nn_labels == nn_id)
    nn_area = int(nn_mask.sum())
    if nn_area == 0:
        return rr_mask, "RR (empty NN region)", 0.0

    overlap = int((rr_mask & nn_mask).sum())
    frac = overlap / float(nn_area)

    if frac >= nn_min_frac:
        return rr_mask, f"RR (frac={frac:.2f} ≥ {nn_min_frac:.2f})", frac

    if below_mode == "NN":
        return nn_mask, f"NN (frac={frac:.2f} < {nn_min_frac:.2f}, favor NN)", frac
    else:
        return rr_mask, f"RR (frac={frac:.2f} < {nn_min_frac:.2f}, favor RR)", frac


# ============================
# MAIN FUNCTION
# ============================

def run_S1_regiongrow(
    CHANNEL,
    VIEW_DIR,
    S1_8,
    S1_9,
    S11_8,
    S11_9,
    nn_min_frac_initial=0.80,
    below_mode_initial="NN",
):

    stack, zlist, flist = load_stack_sorted(VIEW_DIR, f"{CHANNEL}_ch2_")
    pairs = list(zip(zlist, flist))

    out_dir = S1_8 if CHANNEL == "800" else S1_9
    nn_dir  = S11_8 if CHANNEL == "800" else S11_9
    os.makedirs(out_dir, exist_ok=True)

    out_json   = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.json")
    out_csv    = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.csv")
    out_overlay = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_overlay_Z{z:03d}.png")
    out_lblnpy  = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_labels_Z{z:03d}.npy")

    centers = {}

    # Initialize empty label maps using first image size
    sample = cv2.imread(pairs[0][1], cv2.IMREAD_GRAYSCALE)
    label_maps = {z: np.zeros_like(sample, np.int32) for z, _ in pairs}

    active_mask = {z: None for z, _ in pairs}
    first_seed = {z: None for z, _ in pairs}

    nn_min_frac = float(nn_min_frac_initial)
    below_mode  = below_mode_initial
    last_frac   = 1.0

    i = 0
    win = "RegionGrow+NN (RAW | NN | FINAL)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        nonlocal i, nn_min_frac, below_mode, last_frac

        if event == cv2.EVENT_LBUTTONDOWN:
            z, rpath = pairs[i]

            raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
            if raw is None:
                return

            raw_proc = cv2.GaussianBlur(raw, (3, 3), 0)

            lbl = label_maps[z]
            r, c = int(y), int(x)

            if active_mask[z] is None or not active_mask[z][r, c]:

                region_rr = adaptive_region_grow(
                    raw_proc,
                    (r, c),
                    rel_drop=0.6,
                    max_radius=25,
                    mask_existing=lbl,
                )

                if np.any(region_rr):

                    nn_path = os.path.join(nn_dir, f"Z{z:03d}_nn_labels.npy")

                    if os.path.exists(nn_path):
                        nn_labels = np.load(nn_path)

                        final_region, src, frac = choose_rr_or_nn(
                            rr_mask=region_rr.astype(bool),
                            nn_labels=nn_labels,
                            seed_rc=(r, c),
                            nn_min_frac=nn_min_frac,
                            below_mode=below_mode,
                        )

                        last_frac = frac
                        print(f"[Z{z:03d}] {src} | overlap={frac:.2f}")

                    else:
                        final_region = region_rr.astype(bool)
                        last_frac = 1.0
                        print(f"[Z{z:03d}] RR only (no NN file)")

                    new_id = int(lbl.max()) + 1
                    lbl[final_region] = new_id
                    active_mask[z] = final_region
                    first_seed[z] = (r, c)
                    centers.setdefault(z, []).append((r, c))

        elif event == cv2.EVENT_RBUTTONDOWN:
            z, _ = pairs[i]
            lbl = label_maps[z]
            if lbl.max() > 0:
                lbl[lbl == lbl.max()] = 0
                if z in centers and centers[z]:
                    centers[z].pop()

            active_mask[z] = None
            first_seed[z] = None
            print(f"[Undo] Z{z:03d} last region removed.")

    cv2.setMouseCallback(win, mouse_cb)

    def make_tripanel(raw, labels_rr, z):

        raw8 = normalize_to_8u(raw)
        raw_rgb = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)

        rr_overlay = colored_overlay_from_labels(labels_rr, raw8)

        nn_path = os.path.join(nn_dir, f"Z{z:03d}_nn_labels.npy")
        if os.path.exists(nn_path):
            nn_labels = np.load(nn_path)
            nn_overlay = colored_overlay_from_labels(nn_labels, raw8)
            nn_ncell = int(nn_labels.max())
        else:
            nn_overlay = raw_rgb.copy()
            nn_ncell = 0

        vis = cv2.hconcat([raw_rgb, nn_overlay, rr_overlay])

        header = (
            f"{CHANNEL} Z{z:03d} | RR={int(labels_rr.max())} | "
            f"NN={nn_ncell} | thr={nn_min_frac:.2f} | mode={below_mode}"
        )
        cv2.putText(vis, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        cv2.putText(
            vis,
            f"Last RR/NN overlap = {last_frac:.2f}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        return vis


    while True:
        z, rpath = pairs[i]
        raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)

        if raw is None:
            break

        vis = make_tripanel(raw, label_maps[z], z)
        cv2.imshow(win, vis)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') and i < len(pairs) - 1:
            i += 1

        elif key == ord('b') and i > 0:
            i -= 1

        elif key == ord('n'):
            below_mode = "NN"
            print("[MODE] favor NN")

        elif key == ord('r'):
            below_mode = "RR"
            print("[MODE] favor RR")

        elif key == ord(']'):
            nn_min_frac = min(0.95, nn_min_frac + 0.05)
            print(f"[THR] {nn_min_frac:.2f}")

        elif key == ord('['):
            nn_min_frac = max(0.50, nn_min_frac - 0.05)
            print(f"[THR] {nn_min_frac:.2f}")

        elif key == ord('s'):
            lbl = label_maps[z]
            raw8 = normalize_to_8u(raw)
            overlay = colored_overlay_from_labels(lbl, raw8)

            cv2.imwrite(out_overlay(z), overlay)
            np.save(out_lblnpy(z), lbl)

            with open(out_json, "w") as f:
                json.dump(centers, f, indent=2)

            df = pd.DataFrame(
                [(CHANNEL, zz, j, r, c)
                 for zz, plist in centers.items()
                 for j, (r, c) in enumerate(plist)],
                columns=["channel", "Z", "point_idx", "row", "col"]
            )
            df.to_csv(out_csv, index=False)
            print(f"[Saved] Z{z:03d}")

        elif key == 27:
            break

    cv2.destroyAllWindows()
