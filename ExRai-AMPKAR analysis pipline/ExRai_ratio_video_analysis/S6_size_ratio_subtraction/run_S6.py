

# ==========================================
# S6: Area Ratio Viewer
# ==========================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from common_functions.display_utils import build_composite_best

# ============================
# MAIN FUNCTION
# ============================

def run_S6_size_ratio(res, mask_map_800, mask_map_920, save_path=None):
    """
    Launch an interactive viewer for visual validation of 800↔920 matched pairs.

    Steps
    -----
    1. Load cross-channel area/intensity ratio results (CSV from S5 or S4).
    2. Display composite overlays with adjustable ratio acceptance threshold.
    3. Allow toggling of 920-channel visibility.
    4. Enable export of accepted pairs to a filtered mapping CSV.

    Parameters
    ----------
    results_csv : str
        Path to CSV file (e.g., output from `compute_area_ratios()`).
    mask_map_800, mask_map_920 : dict
        Mapping of track IDs → mask stack paths for both channels.
    save_dir : str
        Directory where filtered map (and optionally overlays) will be saved.

    Outputs
    -------
    - Interactive widget in Jupyter for threshold tuning.
    - CSV: `{save_dir}/Accepted_pairs.csv` with accepted mappings.
    - Composite overlay preview for visual verification.
    """
    ratio_slider = widgets.FloatSlider(value=2.0, min=1.0, max=10.0, step=0.5,
                                       description="Max Ratio", readout_format=".1f")
    show_920_toggle = widgets.Checkbox(value=True, description="Show 920 channel")
    save_btn = widgets.Button(description="💾 Save Filtered Map", button_style="success")
    out = widgets.Output()

    accepted_global = []

    def update(change=None):
        nonlocal accepted_global
        max_ratio = ratio_slider.value
        show_920 = show_920_toggle.value
        canvas, accepted, rejected = build_composite_best(res, mask_map_800, mask_map_920,
                                                          max_ratio=max_ratio, show_920=show_920)
        accepted_global = accepted

        with out:
            clear_output(wait=True)
            plt.figure(figsize=(8,8))
            plt.imshow(canvas)
            plt.axis("off")
            plt.title(f"Accepted: {len(accepted)} | Rejected: {len(rejected)} (threshold={max_ratio:.1f})")
            plt.show()

    def on_save(_):
        if not accepted_global:
            print("⚠️ No accepted pairs to save!")
            return
        df_out = pd.DataFrame([{"track_800": t800, "track_920": t920, "ratio": ratio}
                               for t800, t920, ratio in accepted_global])
        if save_path:
            df_out.to_csv(save_path, index=False)
            print(f"💾 Saved {len(df_out)} filtered pairs to {save_path}")
        else:
            print(df_out.head())

    ratio_slider.observe(update, names="value")
    show_920_toggle.observe(update, names="value")
    save_btn.on_click(on_save)

    display(widgets.VBox([ratio_slider, show_920_toggle, save_btn, out]))
    update()
