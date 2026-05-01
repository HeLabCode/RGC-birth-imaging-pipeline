import os, json
import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import interp1d

# ---------- 2D similarity transform  ----------

def _fit_similarity(P, Q):
    """
    Estimate 2D similarity transform (translation, rotation, scale) aligning points P → Q.

    Uses OpenCV’s `estimateAffinePartial2D` with LMedS for robustness.
    Input:
        P, Q : array-like of shape (N,2)
            Source and target coordinates (at least 2–3 points recommended).
    Returns:
        A : ndarray (2×3)
            Estimated affine matrix mapping P → Q.
    """
    A, _ = cv2.estimateAffinePartial2D(np.asarray(P, float), np.asarray(Q, float), method=cv2.LMEDS)
    return A



# ---------- Decompose affine matrix ----------

def _decompose_sim(A):
    """
    Decompose a 2×3 similarity affine matrix into translation, rotation, and scale.

    Input:
        A : ndarray (2×3)
            Affine transform as returned by `cv2.estimateAffinePartial2D`.

    Returns:
        tx, ty : translations (float)
        theta  : rotation angle (radians)
        s      : uniform scale factor (float)
    """
    a, b, tx = A[0]
    c, d, ty = A[1]
    s = (np.hypot(a, c) + np.hypot(b, d)) / 2.0
    theta = np.arctan2(c, a)
    return tx, ty, theta, s



# ---------- Compose affine matrix ----------

def _compose_sim(tx, ty, theta, s):
    """
    Compose a 2×3 similarity transform matrix from translation, rotation, and scale.

    Input:
        tx, ty : translations
        theta  : rotation angle (radians)
        s      : uniform scale factor

    Returns:
        A : ndarray (2×3)
            Affine matrix representing the transform.
    """
    c, sn = np.cos(theta), np.sin(theta)
    a, b = s * c, -s * sn
    c2, d = s * sn, s * c
    return np.array([[a, b, tx],
                     [c2, d, ty]], float)



# ---------- Apply affine matrix ----------

def _apply_A(A, x, y):
    """
    Apply a 2×3 affine transform to point coordinates.

    Input:
        A : ndarray (2×3)
            Affine matrix from `_compose_sim` or `_fit_similarity`.
        x, y : float or ndarray
            Coordinates in the source frame.

    Returns:
        x', y' : transformed coordinates (float or ndarray)
    """
    return (A[0,0]*x + A[0,1]*y + A[0,2],
            A[1,0]*x + A[1,1]*y + A[1,2])



# ---------- Estimate geometric transforms from anchor  ----------    
    
def _estimate_perZ_transforms_from_json(channel, z_order, S2_8, S2_9):
    """
    Estimate per-Z slice geometric transforms from 3-point anchor JSON.

    Purpose
    -------
    Aligns each Z-slice of a given channel (800 or 920) to its reference
    coordinate frame by computing 2×3 similarity transforms based on
    manually selected anchor points.

    Steps
    -----
    1. Load the JSON file {channel}_S2_anchors_3pt.json from S2_8 or S2_9.
    2. Choose the first slice with valid 3 anchors as reference (ref_z).
    3. Fit a similarity transform for each slice with 3 anchors:
         (translation, rotation, scale)
    4. Interpolate missing slices along Z for smooth transforms.
    5. Return the full mapping: z → 2×3 affine matrix and ref_z.

    Parameters
    ----------
    channel : str
        "800" or "920" wavelength channel.
    z_order : list[int]
        Ordered list of Z indices to process.

    Returns
    -------
    (dict, int)
        A dictionary mapping each Z → 2×3 similarity matrix,
        and the reference Z index (ref_z).

    Raises
    ------
    FileNotFoundError : if the anchors JSON is missing.
    RuntimeError : if no valid slices with 3 anchors exist.
    """
    """Return dict z-> 2x3 similarity mapping slice(z) coords -> channel REF coords."""
    s2_dir = S2_8 if channel=="800" else S2_9
    anchors_json = os.path.join(s2_dir, f"{channel}_S2_anchors_3pt.json")
    if not os.path.exists(anchors_json):
        raise FileNotFoundError(f"Missing anchors JSON for {channel}: {anchors_json}")
    with open(anchors_json, "r") as f:
        anchors = {int(k): [tuple(p) for p in v] for k,v in json.load(f).items()}
    avail = [z for z in z_order if (z in anchors and len(anchors[z])==3)]
    if not avail:
        raise RuntimeError(f"No slices with 3 anchors found for {channel}")
    ref_z = avail[0]
    Q = np.array(anchors[ref_z], float)

    zs_with = [z for z in z_order if z in anchors and len(anchors[z])==3]
    params = {}
    for z in zs_with:
        A = _fit_similarity(anchors[z], Q)
        params[z] = _decompose_sim(A)

    all_params = {}
    if len(zs_with) == 1:
        for z in z_order:
            all_params[z] = params[zs_with[0]]
    else:
        zs = np.array(zs_with, float)
        arrs = [np.array([params[z][i] for z in zs_with]) for i in range(4)]
        fs = [interp1d(zs, arr, kind="linear", fill_value="extrapolate", assume_sorted=True) for arr in arrs]
        for z in z_order:
            all_params[z] = tuple(float(f(float(z))) for f in fs)
    return {z: _compose_sim(*all_params[z]) for z in z_order}, ref_z

