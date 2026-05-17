# ============================
# S4: Dual Illumination Filtering
# ============================

import os
import numpy as np
import pandas as pd
from skimage import io
from common_functions.data_utils import load_raw_map, load_final_mapping
from S3_mapping.mapping_utils import compute_centroids_from_masks
from S4_illumination_drop_out.illumination_utils import interactive_final_viewer_with_dual_masks


# ============================
# MAIN FUNCTION
# ============================

def run_S4_illumination(VIEW_DIR, S2_8, S2_9, S3, S4):
    """
    Run the illumination-based mapping validation (S4).

    Loads 800/920 stacks, centroids, and final mapping; then opens an
    interactive dual viewer to refine pairs based on illumination masks.
    """
    raw800_map, z800_list, _ = load_raw_map("800", VIEW_DIR)
    raw920_map, z920_list, _ = load_raw_map("920", VIEW_DIR)

    stack800 = np.stack([io.imread(raw800_map[z]) for z in z800_list]).astype(np.float32)
    stack920 = np.stack([io.imread(raw920_map[z]) for z in z920_list]).astype(np.float32)

    cent8_list = [compute_centroids_from_masks("800", z, VIEW_DIR, S2_8, S2_9) for z in z800_list]
    cent8_list = [df for df in cent8_list if df is not None and not df.empty and not df.isna().all(axis=None)]
    cent8_all = pd.concat(cent8_list, ignore_index=True) if cent8_list else pd.DataFrame()

    cent9_list = [compute_centroids_from_masks("920", z, VIEW_DIR, S2_8, S2_9) for z in z920_list]
    cent9_list = [df for df in cent9_list if df is not None and not df.empty and not df.isna().all(axis=None)]
    cent9_all = pd.concat(cent9_list, ignore_index=True) if cent9_list else pd.DataFrame()

    mapping = load_final_mapping(os.path.join(S3, "Matching_map.csv"))

    interactive_final_viewer_with_dual_masks(stack800, stack920,
                                             z800_list, z920_list,
                                             cent8_all, cent9_all,
                                             mapping, S4)
