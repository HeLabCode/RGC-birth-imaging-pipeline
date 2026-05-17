# ==========================================
# S6: Area Ratio Viewer
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from skimage import io
from S6_size_ratio_subtraction.size_utils import get_best_slice_from_maskstack

# ============================
# MAIN FUNCTION
# ============================

def run_S6_size_ratio(res, mask_map_800, mask_map_920, raw_stack_800, raw_stack_920, save_path=None):
    """
    Launch an interactive viewer for visual validation of 800↔920 matched pairs.
    Displays one best-intensity Z slice per cell.
    """
    ratio_slider = widgets.FloatSlider(value=2.0, min=1.0, max=10.0, step=0.5,
                                       description="Max Ratio", readout_format=".1f")
    show_920_toggle = widgets.Checkbox(value=True, description="Show 920 channel")
    save_btn = widgets.Button(description="💾 Save Filtered Map", button_style="success")
    out = widgets.Output()

    accepted_global = []

    def make_canvas(max_ratio, show_920):
        first_stack = io.imread(next(iter(mask_map_800.values())))
        H, W = first_stack.shape[-2], first_stack.shape[-1]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        accepted = []
        rejected = []

        for r in res.itertuples(index=False):
            t800 = int(r.track_800)
            t920 = int(r.track_920)
            ratio = float(r.ratio_symmetric)

            if t800 not in mask_map_800 or t920 not in mask_map_920:
                continue

            mask800, _, _ = get_best_slice_from_maskstack(mask_map_800[t800], raw_stack_800)
            mask920, _, _ = get_best_slice_from_maskstack(mask_map_920[t920], raw_stack_920)

            if mask800 is None or mask920 is None:
                continue
            
            area800 = int(mask800.sum())
            area920 = int(mask920.sum())

            accepted_pair = np.isfinite(ratio) and ratio <= max_ratio

            if accepted_pair:
                accepted.append((t800, t920, ratio))

                draw_items = [
                    (area800, mask800.astype(bool), [0, 0, 255]),      # 800 blue
                    (area920, mask920.astype(bool), [0, 255, 0]),      # 920 green
                ]

                if not show_920:
                    draw_items = draw_items[:1]

            else:
                rejected.append((t800, t920, ratio))

                draw_items = [
                    (area800, mask800.astype(bool), [255, 0, 0]),      # rejected 800 red
                    (area920, mask920.astype(bool), [0, 100, 0]),      # rejected 920 dark green
                ]

            if not show_920:
                draw_items = draw_items[:1]

# draw larger first, smaller last, so smaller stays visible
            for _, mask, color in sorted(draw_items, key=lambda x: x[0], reverse=True):
                canvas[mask] = color

        return canvas, accepted, rejected

    def update(change=None):
        nonlocal accepted_global
        max_ratio = ratio_slider.value
        show_920 = show_920_toggle.value

        canvas, accepted, rejected = make_canvas(max_ratio, show_920)
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
