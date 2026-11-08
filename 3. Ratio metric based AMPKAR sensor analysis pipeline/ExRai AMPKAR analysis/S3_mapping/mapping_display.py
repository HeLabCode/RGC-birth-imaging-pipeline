import ipywidgets as widgets
from IPython.display import display
import pandas as pd


def interactive_consensus_editor(per800, mapping_init, save_path=None, n_cols=7):
    """
    Build an interactive dropdown interface to manually review and edit the
    800→920 mapping after automatic consensus matching.

    Each row corresponds to one 800 nm track and lists its most likely 920 nm
    matches (based on vote counts). The user can manually select or override
    matches and save the final table to CSV.

    Parameters
    ----------
    per800 : dict[int, Counter]
        Per-track candidate matches and their vote counts.
    mapping_init : dict[int, int]
        Initial 800→920 mapping used to pre-fill dropdowns.
    save_path : str, optional
        Path to save the final mapping CSV (default: None).
    n_cols : int, optional
        Number of columns for arranging dropdowns in the grid layout.

    Returns
    -------
    editors : dict[int, ipywidgets.Dropdown]
        Mapping of 800-track IDs to their corresponding dropdown widgets.
    """
    editors, items = {}, []
    for t800 in sorted(per800.keys()):
        cnts = per800[t800]
        options = [("— None —", None)] + [(f"{t920} (#{votes})", t920) for t920, votes in cnts.most_common()]
        default = mapping_init.get(t800, options[0][1] if options else None)

        dd = widgets.Dropdown(options=options, value=default, layout=widgets.Layout(width="150px"))
        editors[t800] = dd
        items.append(widgets.HBox([
            widgets.Label(value=f"800:{t800} →", layout=widgets.Layout(width="80px")),
            dd
        ]))

    out = widgets.Output()

    def on_save(_):
        final_map = {t800: dd.value for t800, dd in editors.items() if dd.value is not None}
        with out:
            out.clear_output()
            print("✅ Final mapping:")
            print(" | ".join([f"800:{k}→920:{v}" for k, v in final_map.items()]))
        if save_path:
            pd.DataFrame([{"track_800": k, "track_920": v} for k, v in final_map.items()])\
                .to_csv(save_path, index=False)
            with out:
                print(f"📁 Saved")

    save_btn = widgets.Button(description="Save Final Map", button_style="success")
    save_btn.on_click(on_save)
    grid = widgets.GridBox(items,
        layout=widgets.Layout(grid_template_columns=" ".join(["auto"] * n_cols),
                              grid_gap="5px 10px")
    )
    display(widgets.VBox([grid, save_btn, out]))
    return editors
    
    
    
# ---------- Mapping printer ----------

def print_mapping(mapping: dict, label="Mapping"):
    """
    Nicely format and print a dictionary of 800→920 mappings.

    Prints each 800 nm track alongside its assigned 920 nm partner.
    Used mainly for quick inspection of intermediate or final results.

    Parameters
    ----------
    mapping : dict[int, int]
        Dictionary mapping 800→920 track IDs.
    label : str, optional
        Label prefix for the printed mapping (default: "Mapping").
    """
    if not mapping:
        print(f"{label}: {{}}")
        return
    sorted_items = sorted(mapping.items(), key=lambda x: x[0])
    formatted = ", ".join(f"{k}: {v}" for k, v in sorted_items)
    print(f"{label}: {{ {formatted} }}")

