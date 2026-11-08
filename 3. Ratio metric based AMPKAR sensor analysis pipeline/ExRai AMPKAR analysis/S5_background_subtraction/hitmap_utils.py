import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
from common_functions.data_utils import numeric_cell_key


# -------- Heatmap generation -------- 

def plot_bgsub_heatmap(csv_path, out_png=None, title=None, max_xticks=20, max_yticks=30):
    """
    Visualize per-cell background-subtracted intensities as a heatmap.

    This function loads an S5 background-subtracted CSV file
    (e.g., `*_S5_means_bgsub.csv`) and plots a 2D heatmap
    where each row corresponds to a Z-slice and each column
    corresponds to a tracked cell.

    Color encodes the mean intensity after local background subtraction,
    helping identify brightness variations across Z and between cells.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing background-subtracted intensities.
        Must include a 'z_number' column and one or more 'cell_*' columns.
    out_png : str, optional
        If provided, saves the heatmap to this file (PNG format).
        If None, the plot is only displayed.
    title : str, optional
        Title for the plot. Defaults to the CSV filename if not given.
    max_xticks : int, default=20
        Maximum number of cell tick labels to show on the x-axis
        (to prevent overcrowding).
    max_yticks : int, default=30
        Maximum number of Z tick labels to show on the y-axis.

    Output
    ------
    - Displays the heatmap interactively.
    - Optionally saves it as a PNG file.
    
    """
    df = pd.read_csv(csv_path)
    if "z_number" not in df.columns:
        raise ValueError(f"'z_number' column not found in {csv_path}")

    df = df.sort_values("z_number").reset_index(drop=True)

    cell_cols = [c for c in df.columns if c.startswith("cell_")]
    cell_cols = sorted(cell_cols, key=numeric_cell_key)

    mat = df[cell_cols].fillna(0).to_numpy(dtype=float)

    h = max(2.5, min(8, 0.01 * mat.shape[0])+1)
    w = max(7, min(18, 0.1 * mat.shape[1] + 4))
    plt.figure(figsize=(w, h))

    im = plt.imshow(mat, aspect='auto', origin='lower') 
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('mean intensity')

    plt.title(title or os.path.basename(csv_path))
    plt.ylabel('Z')
    plt.xlabel('Cell ID')

    n_cells = len(cell_cols)
    if n_cells <= max_xticks:
        xticks = np.arange(n_cells)
        xlabels = [re.sub(r'^cell_', '', c) for c in cell_cols]
    else:
        step = int(np.ceil(n_cells / max_xticks))
        xticks = np.arange(0, n_cells, step)
        xlabels = [re.sub(r'^cell_', '', cell_cols[i]) for i in xticks]
    plt.xticks(xticks, xlabels, rotation=90, fontsize=8)

    zvals = df["z_number"].to_numpy()
    n_z = len(zvals)
    if n_z <= max_yticks:
        yticks = np.arange(n_z)
        ylabels = [str(int(z)) for z in zvals]
    else:
        step = int(np.ceil(n_z / max_yticks))
        yticks = np.arange(0, n_z, step)
        ylabels = [str(int(zvals[i])) for i in yticks]
    plt.yticks(yticks, ylabels, fontsize=8)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
    plt.show()

