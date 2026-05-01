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
last_frac = 1.0   # stores last RR/NN overlap


def choose_rr_or_nn(rr_mask, nn_labels, seed_rc, nn_min_frac=0.80, below_mode="NN"):
    """
    Decide whether to keep the regiongrow (RR) region or replace it with
    the NN instance overlapping the seed.

    Parameters
    ----------
    rr_mask : 2D bool
        Candidate region from regiongrow.
    nn_labels : 2D int
        NN instance labels for this Z (0 = background).
    seed_rc : (r, c)
        Seed row/col.
    nn_min_frac : float
        Threshold on overlap fraction = area(RR ∩ NN) / area(NN).
    below_mode : {"NN", "RR"}
        Behavior when overlap < nn_min_frac.
        - "NN": favor NN (replace RR with NN)
        - "RR": favor RR (keep RR even if it's smaller)

    Returns
    -------
    final_mask : 2D bool
        Chosen region mask.
    src : str
        Description of the decision.
    frac : float
        Overlap fraction used for decision.
    """
    r0, c0 = seed_rc
    H, W = nn_labels.shape

    # Out of bounds → RR only
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

    overlap = int((rr_mask & nn_mask).sum())
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
    Run the full S1 region-growing segmentation pipeline for a given channel,
    with optional NN-assisted shape correction.

    This is your interactive S1 stage, now with:
      - raw image, NN segmentation, and RR/final labels shown side by side
      - NN used only for cells you manually pick
      - flexible overlap threshold and RR/NN preference

    Controls
    --------
      Left click : grow or split region (as before)
      Right click: undo last region (as before)
      'u'        : undo last region (as before)
      'x'        : clear current Z-slice (as before)
      's'        : save results for current Z-slice
      'q' / 'b'  : navigate forward/backward through slices
      'ESC'      : save all and exit

      NEW:
      'n'        : when overlap < threshold, favor NN (below_mode = "NN")
      'r'        : when overlap < threshold, favor RR (below_mode = "RR")
      '[' / ']'  : decrease / increase NN overlap threshold
                   (nn_min_frac, default 0.80)

    Parameters
    ----------
    CHANNEL : str
        "800" or "920".
    VIEW_DIR : str
        Folder with channel-specific Z stacks (already discovered by S0).
    S1_8, S1_9 : str
        Output folders for final S1 label maps (800 / 920).
        For your new pipeline, you can pass S12_8 / S12_9 here.
    S11_8, S11_9 : str
        Folders where S11 NN results are stored (Z###_nn_labels.npy).
    nn_min_frac_initial : float
        Initial overlap threshold.
    below_mode_initial : {"NN", "RR"}
        Initial behavior when RR overlap < threshold.
    """

    # -------- Configuration  --------
    # Your original loader: 800_ch2_*, 920_ch2_*, etc.
    stack, zlist, flist = load_stack_sorted(VIEW_DIR, f"{CHANNEL}_ch2_")
    pairs = [(z, None, fpath) for z, fpath in zip(zlist, flist)]

    out_dir = S1_8 if CHANNEL == "800" else S1_9
    nn_dir  = S11_8 if CHANNEL == "800" else S11_9
    os.makedirs(out_dir, exist_ok=True)

    out_json   = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.json")
    out_csv    = os.path.join(out_dir, f"{CHANNEL}_regiongrow_centers.csv")
    out_overlay = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_overlay_Z{z:03d}.png")
    out_lblnpy  = lambda z: os.path.join(out_dir, f"{CHANNEL}_regiongrow_labels_Z{z:03d}.npy")

    # -------- Initialization  --------
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            centers = {int(k): v for k, v in json.load(f).items()}
    else:
        centers = {}

    # Label maps for each Z
    label_maps = {
        z: np.zeros_like(cv2.imread(rpath, cv2.IMREAD_GRAYSCALE), np.int32)
        for z, _, rpath in pairs
    }
    active_mask = {z: None for z, _, _ in pairs}
    first_seed = {z: None for z, _, _ in pairs}
    split_contour_visible = {z: None for z, _, _ in pairs}

    # Hybrid controls (stateful within this run)
    nn_min_frac = float(nn_min_frac_initial)
    below_mode  = below_mode_initial  # "NN" or "RR"

    # -------- Interaction  --------
    i = 0
    win = (
        "RegionGrow+NN: click=add/split | u=undo | x=clear | "
        "s=save | q/b nav | n/r NNvsRR | [/]=thr | ESC quit"
    )
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def mouse_cb(event, x, y, flags, param):
        nonlocal i, nn_min_frac, below_mode

        if event == cv2.EVENT_LBUTTONDOWN:
            z, _, rpath = pairs[i]
            raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
            if raw is None:
                return
            raw = cv2.GaussianBlur(raw, (3, 3), 0)
            centers.setdefault(z, [])

            lbl = label_maps[z]
            mask = active_mask[z]
            split_contour_visible[z] = None

            r, c = int(y), int(x)

            # CASE 1: Add new region
            if mask is None or not mask[r, c]:
                region_rr = adaptive_region_grow(
                    raw,
                    (r, c),
                    rel_drop=0.6,
                    max_radius=25,
                    mask_existing=lbl,
                )
                if np.any(region_rr):
                    # Load NN labels for this Z (if available)
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
                        print(f"[Z{z:03d}] new cell: {src}, overlap={frac:.2f}, mode_below='{below_mode}'")
                        )
                        last_frac = frac

                    else:
                        final_region = region_rr.astype(bool)
                        src = "RR (no NN file)"
                        frac = float("nan")
                        print(f"[Z{z:03d}] new cell: {src}")
                        
                    new_id = int(lbl.max()) + 1
                    lbl[final_region] = new_id
                    active_mask[z] = final_region
                    first_seed[z] = (r, c)
                    centers[z].append((r, c))

            # CASE 2: Split existing region
            elif mask is not None and mask[r, c] and first_seed[z] is not None:
                seed1 = first_seed[z]
                seed2 = (r, c)
                ws, contour = watershed_split(mask, seed1, seed2)
                # Remove old label, add two new ones
                old_id = int(lbl[r, c])
                lbl[lbl == old_id] = 0
                lbl[ws == 1] = int(lbl.max()) + 1
                lbl[ws == 2] = int(lbl.max()) + 1
                split_contour_visible[z] = contour
                active_mask[z] = None
                first_seed[z] = None
                centers[z].append((r, c))

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click undo last region on this Z
            z, _, _ = pairs[i]
            lbl = label_maps[z]
            if lbl.max() > 0:
                last_id = int(lbl.max())
                lbl[lbl == last_id] = 0
                if z in centers and centers[z]:
                    centers[z].pop()
            active_mask[z] = None
            first_seed[z] = None
            print(f"[Undo] Z{z:03d} last region removed.")

    cv2.setMouseCallback(win, mouse_cb)

    def make_tripanel(raw, labels_rr, z, channel, nn_dir_local, nn_min_frac_local, below_mode_local):
        """
        Build a side-by-side visualization:
          [0] RAW
          [1] NN overlay
          [2] RR/final overlay
        """
        raw8 = normalize_to_8u(raw)
        raw_rgb = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)

        # RR / final overlay
        rr_overlay = colored_overlay_from_labels(labels_rr, raw8)

        # NN overlay (if available)
        nn_path = os.path.join(nn_dir_local, f"Z{z:03d}_nn_labels.npy")
        if os.path.exists(nn_path):
            nn_labels = np.load(nn_path)
            nn_overlay = colored_overlay_from_labels(nn_labels, raw8)
            nn_ncell = int(nn_labels.max())
        else:
            nn_overlay = raw_rgb.copy()
            nn_ncell = 0

        # horizontal concatenation: RAW | NN | RR
        vis = cv2.hconcat([raw_rgb, nn_overlay, rr_overlay])

        header = (
            f"{channel} Z{z:03d} | RR cells: {int(labels_rr.max())} | "
            f"NN cells: {nn_ncell} | thr={nn_min_frac_local:.2f} | below='{below_mode_local}'"
        )
        cv2.putText(
            vis, header, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
        )
        cv2.putText(
            vis,
            "click=add/split | u=undo | x=clear | s=save | q/b nav | "
            "n/r NNvsRR | [/]=thr | ESC quit",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

        cv2.putText(
        vis,
        f"Last overlap frac = {last_frac:.2f}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

        return vis

    # -------- Main  --------
    while True:
        z, _, rpath = pairs[i]
        raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
        if raw is None:
            print(f"[WARN] Cannot read {rpath}")
            break

        vis = make_tripanel(
            raw,
            label_maps[z],
            z,
            CHANNEL,
            nn_dir,
            nn_min_frac,
            below_mode,
        )
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            if i < len(pairs) - 1:
                i += 1
        elif key == ord('b'):
            if i > 0:
                i -= 1
        elif key == ord('u'):
            z, _, _ = pairs[i]
            lbl = label_maps[z]
            if lbl.max() > 0:
                last_id = int(lbl.max())
                lbl[lbl == last_id] = 0
                if z in centers and centers[z]:
                    centers[z].pop()
            active_mask[z] = None
            first_seed[z] = None
            print(f"[Undo] Z{z:03d} last region removed.")
        elif key == ord('x'):
            z, _, _ = pairs[i]
            label_maps[z][:] = 0
            centers[z] = []
            active_mask[z] = None
            first_seed[z] = None
            print(f"[Clear] Z{z:03d} all regions cleared.")

        # NEW: hybrid controls
        elif key == ord('n'):
            below_mode = "NN"
            print(f"[Hybrid] below-threshold mode = NN (favor NN when overlap < {nn_min_frac:.2f})")
        elif key == ord('r'):
            below_mode = "RR"
            print(f"[Hybrid] below-threshold mode = RR (favor RR when overlap < {nn_min_frac:.2f})")
        elif key == ord(']'):
            nn_min_frac = min(0.95, nn_min_frac + 0.05)
            print(f"[Hybrid] NN overlap threshold = {nn_min_frac:.2f}")
        elif key == ord('['):
            nn_min_frac = max(0.50, nn_min_frac - 0.05)
            print(f"[Hybrid] NN overlap threshold = {nn_min_frac:.2f}")

        elif key == ord('s'):
            # Save for current Z
            z, _, _ = pairs[i]
            labels = label_maps[z]
            raw = cv2.imread(rpath, cv2.IMREAD_ANYDEPTH)
            raw8 = normalize_to_8u(raw)
            overlay = colored_overlay_from_labels(labels, raw8)
            cv2.imwrite(out_overlay(z), overlay)
            np.save(out_lblnpy(z), labels)

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
            print(f"[Saved Z{z:03d}] {len(np.unique(labels)) - 1} cells -> {out_overlay(z)}")

        elif key == 27:  # ESC
            # Save everything and exit
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
