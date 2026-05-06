# ============================
# S3: Cross-Channel Mapping
# ============================

import os
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

from common_functions.data_utils import load_raw_map  
from S3_mapping.mapping_utils import compute_centroids_from_masks, compare_centroids_z, consensus_mapping_auto
from S3_mapping.mapping_display import interactive_consensus_editor, print_mapping


# ============================
# MAIN FUNCTION
# ============================


def run_S3_mapping(z800, z920, max_dist, offset_x, offset_y, method, VIEW_DIR, S2_8, S2_9):
    """
    Run the full S3 cross-channel mapping workflow (800 nm ↔ 920 nm).

    This function integrates automatic centroid matching, consensus mapping,
    and manual interactive correction through a user interface.

    Workflow
    --------
    1. Compare centroids at selected Z-slices between 800 nm and 920 nm stacks.
    2. Automatically build an initial 800→920 mapping using consensus voting.
    3. Display the mapping for manual correction via dropdown menus.
    4. Save the final user-validated mapping to `Matching_map.csv` in the S3 folder.

    Parameters
    ----------
    z800, z920 : int
        Z-indices (slice numbers) to compare between 800 nm and 920 nm stacks.
    max_dist : float
        Maximum allowed distance (in pixels) for centroid matching.
    offset_x, offset_y : float
        Manual X/Y pixel offsets to align 920 nm centroids relative to 800 nm.
    method : str
        Matching method ('Hungarian' or 'Nearest').

    Outputs
    -------
    - Interactive mapping editor (ipywidgets interface)
    - CSV file with the final 800→920 mapping in S3 folder.
    """

    # ---------- Load raw maps ----------
    raw800_map, z800_list, _ = load_raw_map("800",VIEW_DIR)
    raw920_map, z920_list, _ = load_raw_map("920",VIEW_DIR)

    # ---------- Compare & Build Consensus ----------
    mapping = compare_centroids_z(
    z800, z920, raw800_map, raw920_map, VIEW_DIR, S2_8, S2_9,
    max_dist=max_dist, offset_x=offset_x, offset_y=offset_y, method=method
    )



    cent8_all = pd.concat([
        compute_centroids_from_masks("800", z, VIEW_DIR, S2_8, S2_9) for z in z800_list
    ])
    cent9_all = pd.concat([
        compute_centroids_from_masks("920", z, VIEW_DIR, S2_8, S2_9) for z in z920_list
    ])
    
    
    
    per800, init_map = consensus_mapping_auto(
        cent8_all, cent9_all, z800_list, z920_list,
        max_dist=max_dist, offset_x=offset_x, offset_y=offset_y, method=method
    )
    
    # ---------- Display Results ----------
    print_mapping(init_map, label="Updated Consensus Mapping")

    # ---------- Interactive Correction ----------
    DBG = os.path.join(VIEW_DIR, "results")
    S3 = os.path.join(DBG, "S3_mapping_results")
    os.makedirs(S3, exist_ok=True)

    save_path = os.path.join(S3, "Matching_map.csv")

    interactive_consensus_editor(
        per800, init_map, save_path=save_path
    )

