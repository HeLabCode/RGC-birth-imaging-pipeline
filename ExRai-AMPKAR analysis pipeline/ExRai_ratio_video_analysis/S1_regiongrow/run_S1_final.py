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

    if overlap == 0:
        return rr_mask, "RR (0 NN overlap)", 0.0
    
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
    pairs = list(zip(zlist, flist)) 

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

    sample = cv2.imread(pairs[0][1], cv2.IMREAD_GRAYSCALE)
    label_maps = {z: np.zeros_like(sample, np.int32) for z, _ in pairs}

    accepted_ids = {z: set() for z, _ in pairs}


    split_seed  = {z: None for z, _ in pairs}
    split_label = {z: None for z, _ in pairs}


    nn_min_frac = float(nn_min_frac_initial)
    below_mode  = below_mode_initial 
    last_frac   = 1.0               

    # -------- Interaction setup --------
    i = 0  # index into pairs
    win = "S1 RegionGrow+NN (RAW | NN | FINAL)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    
    BAND_H = 110

    def mouse_cb(event, x, y, flags, param):
        nonlocal i, nn_min_frac, below_mode, last_frac

        if event == cv2.EVENT_LBUTTONDOWN:
            z, rpath = pairs[i]

            raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
            if raw is None:
                return

            H, W = raw.shape

            if y < BAND_H:
                return

            y_img = y - BAND_H
            if not (0 <= y_img < H):
                return

            r = int(y_img)
            c = int(x) % W  


            raw_proc = cv2.GaussianBlur(raw, (3, 3), 0)
            lbl = label_maps[z]

            current_id = int(lbl[r, c])

            # ---------------------------------------------------
            # Case 1: click on empty pixel → add region by RR+NN
            # ---------------------------------------------------
            if current_id == 0:
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
                    centers.setdefault(z, []).append((r, c))

                    split_seed[z]  = None
                    split_label[z] = None

            # ---------------------------------------------------
            # Case 2: click on an existing cell → split/merge logic
            # ---------------------------------------------------
            else:

                if split_seed[z] is None:
                    split_seed[z]  = (r, c)
                    split_label[z] = current_id
                    print(f"[Select] Z{z:03d} cell {current_id} selected for split/merge.")

                # -------- second click: decide split vs merge --------
                else:
                    first_id = split_label[z]
                    r1, c1   = split_seed[z]

                    # -------- 2a. second click inside same cell → watershed split --------
                    if current_id == first_id:
                        cell_mask = (lbl == first_id)

                        ws, contour = watershed_split(cell_mask, (r1, c1), (r, c))

      
                        lbl[cell_mask] = 0


                        new_id1 = int(lbl.max()) + 1
                        lbl[ws == 1] = new_id1
                        new_id2 = int(lbl.max()) + 1
                        lbl[ws == 2] = new_id2

                        accepted_ids[z].discard(first_id)
                        centers.setdefault(z, []).append((r, c))

                        print(
                            f"[Split] Z{z:03d} cell {first_id} "
                            f"→ {new_id1} and {new_id2}"
                        )

                    # -------- 2b. second click in another cell → merge two cells --------
                    else:
                        id_a = first_id
                        id_b = current_id
                        main_id  = min(id_a, id_b)
                        other_id = max(id_a, id_b)

                        lbl[lbl == other_id] = main_id
                        accepted_ids[z].discard(other_id)

                        print(
                            f"[Merge] Z{z:03d} cells {id_a} and {id_b} "
                            f"merged into {main_id}"
                        )


                    split_seed[z]  = None
                    split_label[z] = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # -------- Right-click: undo last region on this Z --------
            z, _ = pairs[i]
            lbl = label_maps[z]
            last_id = int(lbl.max())
            if last_id > 0:
                lbl[lbl == last_id] = 0
                accepted_ids[z].discard(last_id)
                if z in centers and centers[z]:
                    centers[z].pop()
                print(f"[Undo] Z{z:03d} last region {last_id} removed.")

            split_seed[z]  = None
            split_label[z] = None
            
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

        tripanel = cv2.hconcat([raw_rgb, nn_overlay, rr_overlay])
        H, W = tripanel.shape[:2]

        # ------------------------------
        # TOP BAND FOR TEXT (no overlay)
        # ------------------------------
        band = np.zeros((BAND_H, W, 3), dtype=np.uint8)
        band[:] = (0, 0, 0)

        header = (
            f"{CHANNEL} Z{z:03d} | RR={int(labels_rr.max())} | "
            f"NN={nn_ncell} | thr={nn_min_frac:.2f} | mode={below_mode}"
        )
        cv2.putText(band, header, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(band,
                    f"Last RR/NN overlap = {last_frac:.2f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(
            band,
            "click=add/split | right-click=undo | q/b=Z | n/r=NNvsRR | [/]=thr | "
            "a=ACCEPT | m=MERGE | c=CLEAR | s=save | ESC=exit",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        vis = np.vstack([band, tripanel])
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

        elif key == ord('a'):
            z, _ = pairs[i]
            lbl = label_maps[z]
            last_id = int(lbl.max())
            if last_id > 0:
                accepted_ids[z].add(last_id)
                print(f"[Accept] Z{z:03d} cell {last_id} marked as ACCEPTED.")
            else:
                print(f"[Accept] Z{z:03d}: no cells to accept.")


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
                
        elif key == ord('c'):
            z, _ = pairs[i]
            label_maps[z][:] = 0
            centers[z] = []
            accepted_ids[z].clear()
            split_seed[z]  = None
            split_label[z] = None
            print(f"[Clear] Z{z:03d} all regions cleared.")

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

