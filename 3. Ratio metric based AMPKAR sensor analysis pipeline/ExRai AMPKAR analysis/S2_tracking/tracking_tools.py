
import cv2
import numpy as np
from scipy.interpolate import interp1d



# --------  Manual anchor collection  --------

def collect_three_anchors(frames_8u, scale=1.0, prefilled=None):
    """
    Collect 3 manual anchor points (A, B, C) per Z-slice for registration.

    Left-click to place 3 anchors per slice. Press ENTER/SPACE to confirm,
    or ESC to reset anchors for the current slice.

    Parameters
    ----------
    frames_8u : list of tuple
        List of (z, 8-bit grayscale image) for each slice.
    scale : float, optional
        Display scale factor for visualization.
    prefilled : dict or None, optional
        Existing anchor dictionary to continue from.

    Returns
    -------
    anchors : dict
        Mapping {z_index: [(x1, y1), (x2, y2), (x3, y3)]}.
    """
    anchors = {} if prefilled is None else {
        int(k): [tuple(p) for p in v] for k, v in prefilled.items()
    }

    for z, img8 in frames_8u:
        if z in anchors and len(anchors[z]) == 3:
            continue

        disp = cv2.resize(img8, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        clicked = []

        def _cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 3:
                clicked.append((x / scale, y / scale))

        win = f"Click 3 anchors (A,B,C) Z{z:03d}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, _cb)

        while True:
            vis = disp.copy()
            for i, (px, py) in enumerate(clicked):
                cx, cy = int(px * scale), int(py * scale)
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    vis, "ABC"[i], (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
                )
            cv2.imshow(win, vis)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32):
                break
            if key == 27:
                clicked = []
                break

        cv2.destroyWindow(win)
        if len(clicked) == 3:
            anchors[z] = [(float(x), float(y)) for (x, y) in clicked]
        else:
            print(f"⚠️ Z{z:03d}: collected {len(clicked)} anchors (need 3)")

    return anchors



# --------  Similarity transform helpers  --------

def _fit_similarity(P, Q):
    """Estimate similarity transform from points P→Q."""
    A, _ = cv2.estimateAffinePartial2D(np.asarray(P, float), np.asarray(Q, float), method=cv2.LMEDS)
    return A

def _decompose(A):
    """Decompose affine matrix into translation, rotation, and scale."""
    a, b, tx = A[0]
    c, d, ty = A[1]
    s = (np.hypot(a, c) + np.hypot(b, d)) / 2.0
    theta = np.arctan2(c, a)
    return tx, ty, theta, s

def _compose(tx, ty, theta, s):
    """Compose affine matrix from translation, rotation, and scale."""
    cos, sin = np.cos(theta), np.sin(theta)
    a, b = s * cos, -s * sin
    c, d = s * sin, s * cos
    return np.array([[a, b, tx], [c, d, ty]], float)

def estimate_transforms(z_order, anchors, ref_z=None):
    """
    Estimate per-Z affine transforms to a reference slice using anchor points.

    Returns
    -------
    A_by_z : dict
        Mapping {z_index: 2×3 affine matrix}.
    """
    if ref_z is None:
        ref_z = z_order[0]
    Q = np.array(anchors[ref_z], float)
    zs_with = [z for z in z_order if z in anchors and len(anchors[z]) == 3]

    params = {z: _decompose(_fit_similarity(anchors[z], Q)) for z in zs_with}
    all_params = {}

    if len(zs_with) == 1:
        for z in z_order:
            all_params[z] = params[zs_with[0]]
    else:
        zs = np.array(zs_with, float)
        arrs = [np.array([params[z][i] for z in zs_with]) for i in range(4)]
        fs = [interp1d(zs, arr, kind="linear", fill_value="extrapolate", assume_sorted=True) for arr in arrs]
        for z in z_order:
            all_params[z] = tuple(float(f(z)) for f in fs)

    A_by_z = {z: _compose(*all_params[z]) for z in z_order}
    return A_by_z
