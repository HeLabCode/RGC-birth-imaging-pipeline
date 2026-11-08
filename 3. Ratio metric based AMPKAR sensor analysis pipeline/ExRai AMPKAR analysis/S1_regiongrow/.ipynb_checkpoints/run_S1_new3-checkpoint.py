# ============================
# S1: Region Growing Segmentation (Hybrid with NN + acceptance/merge)
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
    """
    Decide whether to keep the regiongrow (RR) region or replace it with
    the NN instance overlapping the seed.
    """

    r0, c0 = seed_rc
    H, W = nn_labels.shape

    # Seed out of bounds → RR only
    if not (0 <= r0 < H and 0 <= c0 < W):
        return rr_mask, "RR (seed out of bounds)", 0.0

    nn_id = nn_labels[r0, c0]
    if nn_id == 0:
        # Seed not inside any NN cell
        return rr_mask, "RR (no NN at seed)", 0.0

    nn_mask = (nn_labels == nn_id)
    nn_area = int(nn_mask.sum())
    if nn_area == 0:
        return rr_mask, "RR (empty NN region)", 0.0

    # overlap fraction = how much of the NN cell is captured by RR
    overlap = int((rr_mask & nn_mask).sum())

    if overlap == 0:
        return rr_mask, "RR (0 NN overlap)", 0.0
    
    frac = overlap / float(nn_area)


    if frac >= nn_min_frac:
        return rr_mask, f"RR (frac={frac:.2f} ≥ {nn_min_frac:.2f})", frac

    # Below threshold: behavior depends on below_mode
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
    """
    Interactive S1 segmentation with:
      - Region growing (RR) + optional NN-assisted shape correction
      - RAW | NN | FINAL tripanel display
      - Per-cell acceptance with 'x'
      - Merge all unaccepted cells into one with 'm'

    Controls
    --------
      Mouse:
        Left click     : add new region or split existing region
        Right click    : undo last region on this Z

      Keys:
        q              : next Z
        b              : previous Z
        n              : favor NN when overlap < threshold
        r              : favor RR when overlap < threshold
        [ / ]          : decrease / increase NN overlap threshold
        x              : ACCEPT last-added cell on this Z
        m              : MERGE all UNACCEPTED cells on this Z into one
        c              : CLEAR all cells on this Z
        s              : save overlay + labels + centers/CSV
        ESC            : save global centers/CSV and exit
    """

    # -------- Load Z-stack for this channel --------
    stack, zlist, flist = load_stack_sorted(VIEW_DIR, f"{CHANNEL}_ch2_")
    pairs = list(zip(zlist, flist))  # (z, filepath)

    out_dir = S1_8 if CHANNEL == "800" else S1_9
    nn_dir  = S11_8 if CHANNEL == "800" else S11_9
    os.makedirs(out_dir, exist_ok=True)

    out_json   = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.json")
    out_csv    = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.csv")
    out_overlay = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_overlay_Z{z:03d}.png")
    out_lblnpy  = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_labels_Z{z:03d}.npy")

    # -------- Initialize centers & label maps --------
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            centers = {int(k): v for k, v in json.load(f).items()}
    else:
        centers = {}

    # initialize label maps based on first image size
    sample = cv2.imread(pairs[0][1], cv2.IMREAD_GRAYSCALE)
    label_maps = {z: np.zeros_like(sample, np.int32) for z, _ in pairs}

    active_mask = {z: None for z, _ in pairs}
    first_seed = {z: None for z, _ in pairs}

    # which cells are ACCEPTED (per Z)
    accepted_ids = {z: set() for z, _ in pairs}

    # hybrid controls
    nn_min_frac = float(nn_min_frac_initial)
    below_mode  = below_mode_initial  # "NN" or "RR"
    last_frac   = 1.0                 # last RR/NN overlap fraction

    # -------- Interaction setup --------
    i = 0  # index into pairs
    win = "S1 RegionGrow+NN (RAW | NN | FINAL)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        nonlocal i, nn_min_frac, below_mode, last_frac

        if event == cv2.EVENT_LBUTTONDOWN:
            z, rpath = pairs[i]

            raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
            if raw is None:
                return

            # RAW is for display; RR uses a lightly processed copy
            raw_proc = cv2.GaussianBlur(raw, (3, 3), 0)

            lbl = label_maps[z]
            r, c = int(y), int(x)

            # Add new region
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

            # Split existing region
            elif active_mask[z] is not None and active_mask[z][r, c] and first_seed[z] is not None:
                seed1 = first_seed[z]
                seed2 = (r, c)
                ws, contour = watershed_split(active_mask[z], seed1, seed2)

                old_id = int(label_maps[z][r, c])
                label_maps[z][label_maps[z] == old_id] = 0
                label_maps[z][ws == 1] = int(label_maps[z].max()) + 1
                label_maps[z][ws == 2] = int(label_maps[z].max()) + 1

                # splitting: none of the new ones are auto-accepted
                accepted_ids[z].discard(old_id)
                active_mask[z] = None
                first_seed[z] = None
                centers.setdefault(z, []).append((r, c))

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: undo last region on this Z
            z, _ = pairs[i]
            lbl = label_maps[z]
            last_id = int(lbl.max())
            if last_id > 0:
                lbl[lbl == last_id] = 0
                accepted_ids[z].discard(last_id)
                if z in centers and centers[z]:
                    centers[z].pop()
                print(f"[Undo] Z{z:03d} last region {last_id} removed.")
            active_mask[z] = None
            first_seed[z] = None

    cv2.setMouseCallback(win, mouse_cb)

    def make_tripanel(raw, labels_rr, z):
        raw8 = normalize_to_8u(raw)
        raw_rgb = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)

        # final RR/accepted labels (same labels; acceptance is tracked separately)
        rr_overlay = colored_overlay_from_labels(labels_rr, raw8)

        # NN overlay (if available)
        nn_path = os.path.join(nn_dir, f"Z{z:03d}_nn_labels.npy")
        if os.path.exists(nn_path):
            nn_labels = np.load(nn_path)
            nn_overlay = colored_overlay_from_labels(nn_labels, raw8)
            nn_ncell = int(nn_labels.max())
        else:
            nn_overlay = raw_rgb.copy()
            nn_ncell = 0

        # concat RAW | NN | RR
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

        # instructions, including new ones
        cv2.putText(
            vis,
            "click=add/split | right-click=undo | q/b=Z | n/r=NNvsRR | [/]=thr | "
            "a=ACCEPT last | m=MERGE unaccepted | c=CLEAR Z | s=save | ESC=exit",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        return vis

    # -------- Main interactive loop --------
    while True:
        z, rpath = pairs[i]
        raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
        if raw is None:
            print(f"[WARN] Cannot read {rpath}")
            break

        vis = make_tripanel(raw, label_maps[z], z)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') and i < len(pairs) - 1:
            i += 1

        elif key == ord('b') and i > 0:
            i -= 1

        # --- Hybrid NN/RR behavior controls ---
        elif key == ord('n'):
            below_mode = "NN"
            print(f"[Hybrid] Below-threshold mode = NN (favor NN when overlap < {nn_min_frac:.2f})")
        elif key == ord('r'):
            below_mode = "RR"
            print(f"[Hybrid] Below-threshold mode = RR (favor RR when overlap < {nn_min_frac:.2f})")
        elif key == ord(']'):
            nn_min_frac = min(0.95, nn_min_frac + 0.05)
            print(f"[Hybrid] NN overlap threshold = {nn_min_frac:.2f}")
        elif key == ord('['):
            nn_min_frac = max(0.50, nn_min_frac - 0.05)
            print(f"[Hybrid] NN overlap threshold = {nn_min_frac:.2f}")

        # --- NEW: ACCEPT last-added cell on this Z ---
        elif key == ord('a'):
            z, _ = pairs[i]
            lbl = label_maps[z]
            last_id = int(lbl.max())
            if last_id > 0:
                accepted_ids[z].add(last_id)
                print(f"[Accept] Z{z:03d} cell {last_id} marked as ACCEPTED.")
            else:
                print(f"[Accept] Z{z:03d}: no cells to accept.")

        # --- NEW: MERGE all UNACCEPTED cells on this Z into one ---
        elif key == ord('m'):
            z, _ = pairs[i]
            lbl = label_maps[z]
            all_ids = [cid for cid in np.unique(lbl) if cid > 0]
            unaccepted = [cid for cid in all_ids if cid not in accepted_ids[z]]

            if len(unaccepted) <= 1:
                print(f"[Merge] Z{z:03d}: nothing to merge (0 or 1 unaccepted cell).")
            else:
                main_id = min(unaccepted)
                for cid in unaccepted:
                    if cid == main_id:
                        continue
                    lbl[lbl == cid] = main_id
                print(
                    f"[Merge] Z{z:03d}: merged {len(unaccepted)} unaccepted cells "
                    f"into ID {main_id}."
                )

        # --- CLEAR current Z (new key) ---
        elif key == ord('c'):
            z, _ = pairs[i]
            label_maps[z][:] = 0
            centers[z] = []
            accepted_ids[z].clear()
            active_mask[z] = None
            first_seed[z] = None
            print(f"[Clear] Z{z:03d} all regions cleared.")

        # --- Save current Z ---
        elif key == ord('s'):
            z, _ = pairs[i]
            lbl = label_maps[z]
            raw8 = normalize_to_8u(raw)
            overlay = colored_overlay_from_labels(lbl, raw8)

            cv2.imwrite(out_overlay(z), overlay)
            np.save(out_lblnpy(z), lbl)

            with open(out_json, "w") as f:
                json.dump({int(k): v for k, v in centers.items()}, f, indent=2)

            df = pd.DataFrame(
                [
                    (CHANNEL, zz, j, r, c)
                    for zz, plist in centers.items()
                    for j, (r, c) in enumerate(plist)
                ],
                columns=["channel", "Z", "point_idx", "row", "col"],
            )
            df.to_csv(out_csv, index=False)
            print(f"[Saved] Z{z:03d} | cells={int(lbl.max())}")

        # --- ESC: save centers/CSV and exit ---
        elif key == 27:  # ESC
            with open(out_json, "w") as f:
                json.dump({int(k): v for k, v in centers.items()}, f, indent=2)
            df = pd.DataFrame(
                [
                    (CHANNEL, zz, j, r, c)
                    for zz, plist in centers.items()
                    for j, (r, c) in enumerate(plist)
                ],
                columns=["channel", "Z", "point_idx", "row", "col"],
            )
            df.to_csv(out_csv, index=False)
            print("[Exit] Data saved.")
            break

    cv2.destroyAllWindows()

