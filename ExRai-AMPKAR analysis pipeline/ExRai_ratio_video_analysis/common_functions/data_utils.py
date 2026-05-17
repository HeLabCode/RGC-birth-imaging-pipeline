import os
import re
import pandas as pd
import glob
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import cv2
from skimage import measure
from common_functions.display_utils import print_safe
from common_functions.geometry_utils import _apply_A
    
    
# -------- Image normalization utility -------- 

def normalize_to_8u(img):
    """
    Normalize an image to 8-bit range [0, 255] using robust percentiles.

    The image is rescaled based on the 1st and 99th intensity percentiles
    to suppress outliers and improve contrast. Values are then clipped
    to [0, 1] and converted to 8-bit unsigned integers.

    Parameters
    ----------
    img : ndarray
        Input image as a NumPy array of any numeric dtype.

    Returns
    -------
    img_8u : ndarray
        Image normalized and converted to 8-bit (uint8) format.
    """
    img = img.astype(np.float32)
    vmax = np.percentile(img, 99)
    vmin = np.percentile(img, 1)
    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)



# -------- Image stack loading utility -------- 

def load_stack_sorted(view_dir, prefix):
    """
    Return (stack_list, z_indices, file_list) sorted by numeric Z.

    This function searches for all .tif images in the given directory whose
    filenames match the pattern '<prefix>*Z*.tif'. It extracts the numeric Z
    index from each filename, sorts the files numerically by this index, and
    loads them into memory as a list of 2D image arrays.

    Parameters
    ----------
    view_dir : str
        Directory containing the image files.
    prefix : str
        Filename prefix used to identify the image stack (e.g., '800_').

    Returns
    -------
    stack_list : list of ndarray
        List of 2D image arrays loaded from disk, sorted by Z index.
    z_indices : list of int
        Numeric Z values extracted from filenames.
    file_list : list of str
        Corresponding sorted file paths.
    """
    files = glob.glob(os.path.join(view_dir, f"{prefix}*Z*.tif"))
    parsed = []
    for f in files:
        m = re.search(r"[Zz](\d+)", os.path.basename(f))
        if not m: 
            continue
        znum = int(m.group(1))
        parsed.append((znum, f))
    parsed.sort(key=lambda x: x[0])
    zlist = [z for z,_ in parsed]
    flist = [f for _,f in parsed]
    stack = [io.imread(f) for f in flist]
    return stack, zlist, flist



# -------- Interactive Z-stack browser -------- 

def browse_stacks(stack800, z800, stack920, z920, z800_idx=0, z920_idx=0):
    """
    Show one slice from the 800 nm and 920 nm stacks side by side.

    Used to visually compare the same Z-slice from both channels.
    Works best with Jupyter's `interact()` to scroll through slices.

    Parameters
    ----------
    stack800, stack920 : list of ndarray
        Image stacks for the 800 nm and 920 nm channels.
    z800, z920 : list of int
        Z indices for each stack.
    z800_idx, z920_idx : int
        Current slice index to display.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].imshow(stack800[z800_idx], cmap="gray")
    axs[0].set_title(f"800 nm (Z{z800[z800_idx]:03d})")
    axs[0].axis("off")

    axs[1].imshow(stack920[z920_idx], cmap="gray")
    axs[1].set_title(f"920 nm (Z{z920[z920_idx]:03d})")
    axs[1].axis("off")

    plt.show()



# -------- Extract the numeric Z-index --------  

def _parse_z(fname):
    """
    Extract the numeric Z-index from a filename.

    This function searches for patterns like 'Z003', 'z15', etc.
    and returns the integer value (e.g., 3, 15). If no Z pattern
    is found, it returns None.

    Parameters
    ----------
    fname : str
        Path or filename containing a Z-index substring.

    Returns
    -------
    int or None
        Extracted Z-index as an integer, or None if no match found.
    """
    m = re.search(r"[Zz](\d+)", os.path.basename(fname))
    return int(m.group(1)) if m else None



# -------- Load S1 label + raw data -------- 

def load_s1_items(channel, S1_8, S1_9, VIEW_DIR):
    """
    Load labeled masks and corresponding raw images from S1 output.

    Parameters
    ----------
    channel : str
        Channel identifier ("800" or "920").

    Returns
    -------
    items : list of tuple
        Each tuple is (z_index, label_image, raw_image).
    """
    lab_dir = S1_8 if channel == "800" else S1_9
    lab_files = sorted(
        glob.glob(os.path.join(lab_dir, f"{channel}_regiongrow_labels*.npy")),
        key=_parse_z
    )
    raw_map = {
        _parse_z(p): p
        for p in glob.glob(os.path.join(VIEW_DIR, f"{channel}_*Z*.tif"))
    }

    items = []
    for lf in lab_files:
        z = _parse_z(lf)
        labels = np.load(lf).astype(np.int32)
        raw = io.imread(raw_map[z]) if z in raw_map else (labels > 0).astype(np.uint16) * 1000
        items.append((z, labels, raw))
    return items



# -------- Export track -------- 

def export_track_masks_and_index(channel, items, linked, out_dir):
    """
    Export per-track masks, label images, and index CSV.

    For each tracked object (particle), this function saves:
      - A 3D binary mask stack (.tif)
      - Per-Z 2D binary mask images (.tif)
      - A CSV file with per-slice info (centroid, area, bbox, etc.)

    Parameters
    ----------
    channel : str
        Channel name, e.g. "800" or "920".
    items : list of tuple
        List of (z, labels, raw) from previous segmentation steps.
    linked : pandas.DataFrame
        Tracking results containing 'particle', 'label_id', and 'z' columns.
    out_dir : str
        Output directory to save results.

    Returns
    -------
    int
        Number of exported tracks.
    """
    if linked.empty:
        return 0

    z_order = [z for z, _, _ in items]
    z_to_labels = {z: labels for z, labels, _ in items}
    H, W = next(iter(z_to_labels.values())).shape
    particles = sorted(linked["particle"].unique().tolist())

    perZ_trackLabel = {z: np.zeros((H, W), dtype=np.uint16) for z in z_order}
    index_rows = []
    by_z = {z: linked[linked["z"] == z] for z in z_order}

    for z in z_order:
        labels = z_to_labels[z]
        rows_z = by_z.get(z, pd.DataFrame())
        if rows_z.empty:
            continue
        for _, r in rows_z.iterrows():
            pid = int(r["particle"])
            lid = int(r["label_id"])
            mask = (labels == lid)
            perZ_trackLabel[z][mask] = pid + 1

            props = measure.regionprops((labels == lid).astype(np.uint8))
            if props:
                y, x = props[0].centroid
                area = int(props[0].area)
                bbox = props[0].bbox
            else:
                y, x, area, bbox = np.nan, np.nan, 0, (0, 0, 0, 0)

            index_rows.append({
                "channel": channel,
                "track_id": pid,
                "z": int(z),
                "label_id": lid,
                "area_px": area,
                "centroid_y": float(y),
                "centroid_x": float(x),
                "bbox_ymin": int(bbox[0]), "bbox_xmin": int(bbox[1]),
                "bbox_ymax": int(bbox[2]), "bbox_xmax": int(bbox[3])
            })

    idx_df = pd.DataFrame(index_rows).sort_values(["track_id", "z"])
    idx_df.to_csv(os.path.join(out_dir, f"{channel}_S2_track_index.csv"), index=False)

    count = 0
    for pid in particles:
        sub = linked[linked["particle"] == pid]
        stack = np.zeros((len(z_order), H, W), dtype=np.uint8)
        for _, row in sub.iterrows():
            z = int(row["z"])
            lab = int(row["label_id"])
            stack[z_order.index(z)] = (z_to_labels[z] == lab).astype(np.uint8)

        tif_path = os.path.join(out_dir, f"{channel}_track{pid:04d}_maskstack.tif")
        io.imsave(tif_path, (stack * 255).astype(np.uint8), check_contrast=False)

        for zi, z in enumerate(z_order):
            png_path = os.path.join(out_dir, f"{channel}_track{pid:04d}_Z{z:03d}_mask.tif")
            cv2.imwrite(png_path, (stack[zi] * 255).astype(np.uint8))

        count += 1

    return count


# -------- Load raw TIFF Z-stack map --------
def load_raw_map(channel,VIEW_DIR):
    """
    Build a mapping of Z-index → raw .tif image paths for a given channel.

    This function searches the main VIEW_DIR for files matching the pattern
    '{channel}_*Z*.tif', extracts the numeric Z value from each filename, 
    sorts them, and returns:
      - a dictionary of Z-index to file path,
      - the sorted list of Z indices,
      - and the image dimensions (H, W).

    Parameters
    ----------
    channel : str
        Channel name ("800" or "920").

    Returns
    -------
    tuple
        (zmap, zlist, (H, W))
        zmap : dict[int, str]
            Maps Z indices to file paths.
        zlist : list[int]
            Sorted Z indices.
        (H, W) : tuple[int, int]
            Image height and width from the first slice.
    """
    paths = glob.glob(os.path.join(VIEW_DIR, f"{channel}_*Z*.tif"))
    zmap = { _parse_z(p): p for p in paths }
    zlist = sorted(zmap.keys())
    if not zlist:
        raise RuntimeError(f"No raw slices found for channel {channel} under {VIEW_DIR}")

    sample = io.imread(zmap[zlist[0]])
    if sample.ndim == 2:
        H, W = sample.shape
    else:
        H, W = sample.shape[:2]

    return zmap, zlist, (H, W)



# -------- Universal mask stack indexer --------

def index_maskstacks(channel, dir_800, dir_920):
    """
    Build a dictionary linking each track ID to its mask stack file path.
    Works for S2, S3, or any later stage by passing the corresponding
    directories for the 800 nm and 920 nm channels.

    Parameters
    ----------
    channel : str
        Channel name ("800" or "920").
    dir_800 : str
        Directory path for the 800 nm channel results.
    dir_920 : str
        Directory path for the 920 nm channel results.

    Returns
    -------
    dict[int, str]
        Mapping of track ID → absolute file path for each mask stack.
    """
    base_dir = dir_800 if channel == "800" else dir_920
    files = glob.glob(os.path.join(base_dir, f"{channel}_track*_maskstack.tif"))
    out = {}
    pat = re.compile(rf"^{re.escape(channel)}_track(\d+)_maskstack\.tif$", re.IGNORECASE)

    for p in files:
        bn = os.path.basename(p)
        m = pat.match(bn)
        if m:
            tid = int(m.group(1)) 
            out[tid] = p

    return out



# ---------- Load final mapping ----------

def load_final_mapping(csv_path):
    """
    Load a 800→920 cross-channel mapping from CSV into a dictionary.

    This helper reads the mapping file (e.g., 'Matching_map.csv')
    and returns a dictionary suitable for quick lookup.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (usually located in S5 or results folder).

    Returns
    -------
    dict[int, int]
        Mapping from track_800 → track_920 IDs.
    """
    df = pd.read_csv(csv_path)
    if not {"track_800", "track_920"}.issubset(df.columns):
        raise ValueError(f"{csv_path} must contain columns: track_800, track_920")
    return dict(zip(df.track_800.astype(int), df.track_920.astype(int)))


# ---------- Extract track ID ----------

def track_id_from_filename(path):
    """
    Extract the numeric track ID from a mask stack filename.

    Parameters
    ----------
    path : str
        Full path to the per-track mask stack file.

    Returns
    -------
    int
        The integer track ID parsed from the filename.
        
    ValueError
        If the filename does not match the expected pattern
        "_track####_maskstack.tif".
    """
    m = re.search(r"_track(\d+)_maskstack\.tif$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse track id from {path}")
    return int(m.group(1))



# ---------- Extract track ID ----------

def numeric_cell_key(cname: str) -> int:
    """
    Extract numeric cell ID from a column name.

    Example
    -------
    'cell_12' → 12

    Used to sort columns by cell number in heatmap plots.
    """
    m = re.search(r'(\d+)$', cname)
    return int(m.group(1)) if m else 0



# ---------- Reader for combined intensity CSV file. ----------

def _load_intensity_singlefile(path):
    """
    Robust reader for a single combined intensity CSV file.

    Purpose
    -------
    Loads a per-channel intensity file (e.g. `800_S4_means_bgsub.csv`) that may
    exist in either:
      - *Wide format*: columns like ['z_number', 'Cell_1', 'Cell_2', ...]
      - *Long format*: rows with ['track_id', 'z', 'intensity']

    Automatically detects the delimiter (comma, tab, or semicolon), cleans headers,
    and normalizes everything into a long-format DataFrame with standardized
    columns ['track_id', 'z', 'value'].

    Parameters
    ----------
    path : str
        Full path to the intensity CSV file.

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns:
            - track_id : int
                Cell or object ID (derived from 'Cell_X' column name or explicit field).
            - z : int
                Z-slice index.
            - value : float
                Background-subtracted mean intensity for that Z and track.
    """
    def _try_read(_sep, regex=False):
        try:
            if regex:
                return pd.read_csv(path, sep=_sep, engine="python")
            return pd.read_csv(path, sep=_sep, engine="python")
        except Exception:
            return None

    df = _try_read(None)
    if df is None or df.shape[1] == 1:
        df = _try_read("\t") or _try_read(r"[\t,;]+", regex=True)
    if df is None:
        raise ValueError(f"Could not read intensity file: {path}")

    cols = [str(c).strip() for c in df.columns]
    if cols and cols[0].startswith("\ufeff"):
        cols[0] = cols[0].replace("\ufeff", "")
    df.columns = cols

    lower = {c.lower(): c for c in df.columns}
    z_col = None
    for cand in ["z_number", "z", "z_index", "slice", "plane"]:
        if cand in lower:
            z_col = lower[cand]
            break
    cell_cols = [c for c in df.columns if re.fullmatch(r"(?i)cell[\s_]*\d+", str(c).strip())]

    if z_col is not None and len(cell_cols) > 0:
        long = df.melt(id_vars=[z_col], value_vars=cell_cols, var_name="cell", value_name="value")
        long["track_id"] = long["cell"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
        long = long.drop(columns=["cell"]).rename(columns={z_col: "z"})
        long["z"] = pd.to_numeric(long["z"], errors="coerce")
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        long = long.dropna(subset=["z"]).copy()
        long["z"] = long["z"].astype(int)
        print_safe(f"Loaded wide-format intensity from {os.path.basename(path)} "
                    f"→ {long['track_id'].nunique()} cells, Zs {long['z'].min()}..{long['z'].max()}")
        return long[["track_id","z","value"]]

    cols_lower = {c.lower(): c for c in df.columns}
    track_col = next((cols_lower[k] for k in ["track_id","track","cell_id","id"] if k in cols_lower), None)
    z_col     = next((cols_lower[k] for k in ["z","z_number","z_index","slice","plane"] if k in cols_lower), None)
    if track_col is None or z_col is None:
        raise ValueError(
            f"Could not detect S5 wide format or legacy columns in {path}. "
            f"Headers: {list(df.columns)}"
        )

    def _find_value_column(df_):
        for c in ["intensity_bgsub","mean_bgsub","bg_subtracted","bg_corrected","intensity","mean","value"]:
            if c in df_.columns:
                return c
        raise ValueError("No intensity column found in legacy format.")
    val_col = _find_value_column(df)

    out = df[[track_col, z_col, val_col]].copy()
    out.columns = ["track_id","z","value"]
    out["track_id"] = pd.to_numeric(out["track_id"], errors="coerce").astype("Int64").astype(int)
    out["z"] = pd.to_numeric(out["z"], errors="coerce").astype("Int64").astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    print_safe(f"Loaded long-format intensity from {os.path.basename(path)}: {out['track_id'].nunique()} cells.")
    return out



# ---------- Load and merge multiple per-track intensity CSV ----------

def _load_intensity_pertrack(dir_path, channel):
    """
    Load and merge multiple per-track intensity CSV files into a unified long table.

    Purpose
    -------
    Reads all intensity CSVs found in a given directory matching the pattern:
        f"{channel}_track*_intensity_by_z.csv"
    Each file corresponds to one tracked object and contains intensity vs. Z data.

    The function extracts the Z index and intensity column, infers the track ID
    from the filename, and concatenates all files into a single table.

    Parameters
    ----------
    dir_path : str
        Directory containing per-track intensity CSVs.
    channel : str
        Channel identifier ("800" or "920") used in filename matching.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - track_id : int
                Track or cell identifier inferred from filename.
            - z : int
                Z-slice index.
            - value : float
                Background-subtracted mean intensity.
    """
    rows = []
    pats = glob.glob(os.path.join(dir_path, f"{channel}_track*_intensity_by_z.csv"))
    for p in pats:
        
        m = re.search(r"_track(\d+)_", os.path.basename(p))
        if not m:
            continue
        tid = int(m.group(1))
        df = pd.read_csv(p)
        z_col = next((c for c in df.columns if c.lower() in ["z","z_index","slice","plane","z_number"]), None)
        if z_col is None:
            raise ValueError(f"No z column in {p}")
        val_col = next((c for c in df.columns if c.lower() in
                        ["intensity_bgsub","mean_bgsub","bg_subtracted","bg_corrected","intensity","mean","value"]), None)
        if val_col is None:
            raise ValueError(f"No intensity-like column in {p}")
        sub = df[[z_col, val_col]].copy()
        sub.columns = ["z","value"]
        sub["track_id"] = int(tid)
        rows.append(sub[["track_id","z","value"]])
    if not rows:
        raise RuntimeError(f"No per-track intensity CSVs found under {dir_path} for pattern {channel}_track*_intensity_by_z.csv")
    out = pd.concat(rows, ignore_index=True)
    out["z"] = pd.to_numeric(out["z"], errors="coerce").astype("Int64").astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


# ---------- Load per-channel intensity data ----------

def _load_intensity_table(channel, INT_SINGLE_FILE_800, INT_SINGLE_FILE_920, INT_DIR_800, INT_DIR_920):
    """
    Unified interface to load per-channel intensity data.

    Purpose
    -------
    Depending on global configuration, loads the background-corrected intensity
    tables for the specified channel ("800" or "920") from either:
      - a single combined file (wide or long format), or
      - multiple per-track intensity CSV files.

    Parameters
    ----------
    channel : str
        Channel identifier ("800" or "920").

    Returns
    -------
    pandas.DataFrame
        Long-format table with columns:
            - track_id : int
            - z : int
            - value : float

    Raises
    ------
    RuntimeError
        If neither a valid single-file path nor a per-track directory is available.
    ValueError
        If intensity files cannot be parsed correctly.

    """
    single = INT_SINGLE_FILE_800 if channel=="800" else INT_SINGLE_FILE_920
    direc  = INT_DIR_800         if channel=="800" else INT_DIR_920
    if single and os.path.exists(single):
        return _load_intensity_singlefile(single)
    return _load_intensity_pertrack(direc, channel)



# ---------- Manual mapping loader ----------

def load_manual_map(MAPPING_CSV):
    """
    Load the 800→920 mapping table that defines matched cell pairs.

    Purpose
    -------
    Retrieves a manual mapping between tracks in the 800 nm and 920 nm channels.
    The mapping can come from either:
      - a CSV file (path specified by MANUAL_MAP_CSV), or
      - an in-memory Python dictionary (MANUAL_MAP_DICT).

    Parameters
    ----------
    None (uses global configuration variables).

    Returns
    -------
    dict[int, int]
        Mapping of {track_800: track_920} pairs.
    """
    if os.path.exists(MAPPING_CSV):
        df = pd.read_csv(MAPPING_CSV)
        if not set(["track_800", "track_920"]).issubset(df.columns):
            raise ValueError(f"{MAPPING_CSV} must contain columns: track_800, track_920")
        m = {int(r.track_800): int(r.track_920) for r in df.itertuples(index=False)}

        return m
    else:
        raise FileNotFoundError(f"❌ Mapping CSV not found at {MAPPING_CSV}")
        




# ---------- Per-track per-Z centroids (channel REF frame) ----------
def _per_track_centroids_perZ_ref(channel, S2_8, S2_9, z_order, A_by_z, mask_map=None):
    """
    Compute per-track centroids across all Z-slices, transformed into the
    reference coordinate frame defined by per-Z affine similarity transforms.

    Parameters
    ----------
    channel : str
        Channel identifier ("800" or "920").
    S2_8, S2_9 : str
        Paths to the segmentation outputs (used if mask_map is None).
    z_order : list[int]
        Ordered list of Z indices corresponding to raw slices.
    A_by_z : dict[int, np.ndarray]
        Mapping from Z index to affine transform matrix (2×3).
    mask_map : dict[int, str], optional
        Precomputed mapping of track_id → mask stack path. If None,
        mask stacks are auto-indexed using `index_maskstacks`.

    Returns
    -------
    pandas.DataFrame
        Columns: ['track_id', 'z', 'x_ref', 'y_ref']
    """

    if mask_map is None:
        mask_map = index_maskstacks(channel,S2_8,S2_9)
    if not mask_map:
        s2_dir = S2_8 if channel == "800" else S2_9
        raise RuntimeError(f"No mask stacks found for {channel} under {s2_dir}")

    rows = []
    miss = 0
    for tid, p in sorted(mask_map.items()):
        try:
            mstk = io.imread(p)  
        except Exception as e:
            print_safe(f"WARNING: failed to read mask stack {p}: {e}")
            continue
        if mstk.ndim == 2:
            mstk = mstk[np.newaxis, ...]
        mstk = (mstk > 0)

        for zi, z in enumerate(z_order):
            if zi >= mstk.shape[0]:
                break
            mask = mstk[zi]
            if not np.any(mask):
                continue
            rp = measure.regionprops(mask.astype(np.uint8))
            if not rp:
                continue
            y0, x0 = rp[0].centroid  
            A = A_by_z[z]
            x1, y1 = _apply_A(A, x0, y0)  
            rows.append({"track_id": int(tid), "z": int(z), "x_ref": float(x1), "y_ref": float(y1)})

    return pd.DataFrame(rows).sort_values(["track_id","z"]).reset_index(drop=True)
