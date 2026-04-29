
import os
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from S4_illumination_drop_out.build_illumination import build_illumination_mask


# ---------- Interactive viewer ----------
def interactive_final_viewer_with_dual_masks(stack800, stack920,
                                             z800_list, z920_list,
                                             cent8_all, cent9_all,
                                             mapping, S4,
                                             topk=3, radius_px=80):
    """
    Interactive visualization for verifying and refining 800↔920 mapping.

    - Displays illumination masks for both stacks
    - Allows threshold adjustment per channel
    - Marks valid pairs (inside both masks)
    - Saves filtered mapping to CSV
    """
    save_path = os.path.join(S4, "illumination_mapping_filtered_dual.csv")

    keep_slider_800 = widgets.FloatSlider(
        value=0.7, min=0.2, max=0.95, step=0.01,
        description="Keep ≥ 800", readout_format=".2f"
    )
    keep_slider_920 = widgets.FloatSlider(
        value=0.96, min=0.7, max=1.1, step=0.01,
        description="Keep ≥ 920", readout_format=".2f"
    )
    z800_slider = widgets.IntSlider(min=min(z800_list), max=max(z800_list),
                                    value=z800_list[0], description="Z800")
    z920_slider = widgets.IntSlider(min=min(z920_list), max=max(z920_list),
                                    value=z920_list[0], description="Z920")
    save_btn = widgets.Button(description="💾 Save mapping", button_style="success")
    out = widgets.Output()

    kept_pairs = {}

    def update(change=None):
        nonlocal kept_pairs
        z800, z920 = z800_slider.value, z920_slider.value
        keep800, keep920 = keep_slider_800.value, keep_slider_920.value

        mask800, _ = build_illumination_mask(stack800, topk=topk, radius_px=radius_px,
                                             keep_frac=keep800, invert=False)
        mask920, _ = build_illumination_mask(stack920, topk=topk, radius_px=radius_px,
                                             keep_frac=keep920, invert=False)
        

        kept_pairs = {}

        with out:
            clear_output(wait=True)
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            axs[0].imshow(stack800[z800_list.index(z800)], cmap="gray")
            axs[0].imshow(mask800, cmap="Reds", alpha=0.3)
            axs[0].set_title(f"800nm Z={z800} (thr={keep800:.2f})")
            axs[0].axis("off")

            axs[1].imshow(stack920[z920_list.index(z920)], cmap="gray")
            axs[1].imshow(mask920, cmap="Reds", alpha=0.3)
            axs[1].set_title(f"920nm Z={z920} (thr={keep920:.2f})")
            axs[1].axis("off")

            for t800, t920 in mapping.items():
                c8 = cent8_all[(cent8_all.track_id == t800) & (cent8_all.z == z800)]
                c9 = cent9_all[(cent9_all.track_id == t920) & (cent9_all.z == z920)]

                pt800, pt920 = None, None
                valid800, valid920 = True, True

                if not c8.empty:
                    x8, y8 = c8.iloc[0].x_ref, c8.iloc[0].y_ref
                    pt800 = (x8, y8)
                    if not mask800[int(y8), int(x8)]:
                        valid800 = False

                if not c9.empty:
                    x9, y9 = c9.iloc[0].x_ref, c9.iloc[0].y_ref
                    pt920 = (x9, y9)
                    if not mask920[int(y9), int(x9)]:
                        valid920 = False

                valid_pair = valid800 and valid920
                if valid_pair:
                    kept_pairs[t800] = t920

                if pt800 is not None:
                    x, y = pt800
                    if valid_pair:
                        axs[0].plot(x, y, "bo", markersize=6)
                        axs[0].text(x, y-10, str(t800), color="cyan", fontsize=8, ha="center")
                    else:
                        axs[0].plot(x, y, "rx", markersize=8)
                        axs[0].text(x, y-10, str(t800), color="red", fontsize=8, ha="center")

                if pt920 is not None:
                    x, y = pt920
                    if valid_pair:
                        axs[1].plot(x, y, "go", markersize=6)
                        axs[1].text(x, y-10, str(t920), color="lime", fontsize=8, ha="center")
                    else:
                        axs[1].plot(x, y, "rx", markersize=8)
                        axs[1].text(x, y-10, str(t920), color="red", fontsize=8, ha="center")

            plt.tight_layout()
            plt.show()

    # ---------- Full-stack filtering & saving ----------
    def on_save(_):
        keep800, keep920 = keep_slider_800.value, keep_slider_920.value
        mask800, _ = build_illumination_mask(stack800, topk=topk, radius_px=radius_px,
                                            keep_frac=keep800, invert=False)
        mask920, _ = build_illumination_mask(stack920, topk=topk, radius_px=radius_px,
                                            keep_frac=keep920, invert=False)

        kept_pairs_all = {}
        for t800, t920 in mapping.items():
            
            c8s = cent8_all[cent8_all.track_id == t800]
            c9s = cent9_all[cent9_all.track_id == t920]

            valid8 = any(mask800[int(r.y_ref), int(r.x_ref)] for _, r in c8s.iterrows())
            valid9 = any(mask920[int(r.y_ref), int(r.x_ref)] for _, r in c9s.iterrows())

            if valid8 and valid9:
                kept_pairs_all[t800] = t920

        with out:
            print(f"\n💾 Found {len(kept_pairs_all)} valid pairs across the full stack.")

        if kept_pairs_all:
            df = pd.DataFrame([{"track_800": k, "track_920": v}
                            for k, v in kept_pairs_all.items()])
            df.to_csv(save_path, index=False)
            with out:
                print(f"✅ Saved filtered mapping to {save_path}")
        else:
            with out:
                print("⚠️ No pairs within illumination region!")


    save_btn.on_click(on_save)
    for slider in [z800_slider, z920_slider, keep_slider_800, keep_slider_920]:
        slider.observe(update, names="value")

    display(widgets.HBox([z800_slider, z920_slider,
                          keep_slider_800, keep_slider_920,
                          save_btn]), out)
    update()
