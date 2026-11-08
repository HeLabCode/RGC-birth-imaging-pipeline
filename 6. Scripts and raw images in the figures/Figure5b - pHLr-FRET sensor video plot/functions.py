"""Visualization of precomputed per-cell FRET measurements.

Does not perform segmentation, registration, bleed-through correction, or FRET
calculation. Input data should have been independently prepared beforehand.
"""
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tif
from scipy.ndimage import gaussian_filter, distance_transform_edt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable


# ============================================================
# USER-EDITABLE VIDEO AND DISPLAY SETTINGS
# Change these values in functions.py (not in the notebook).
# ============================================================
VIDEO_FPS = 10               # Playback speed in frames/second
VIDEO_DPI = 100              # Video rendering resolution
MOVING_DOT_COLOR = "#00CC44"  # Green cursor dot on the FRET curve
FRET_COLORMAP = "coolwarm"   # Matplotlib cmap name; original custom coolwarm shape retained
FRET_SCALE_MIN = 0.6         # Pseudocolor colorbar lower limit
FRET_SCALE_MAX = 1.0         # Pseudocolor colorbar upper limit
FRET_COLORBAR_TICKS = (0.6, 0.7, 0.8, 0.9, 1.0)

# Internal averaging parameters; keep unchanged for reproducibility.
_FRET_AVERAGING_FRAMES = 7
_DONOR_AVERAGING_FRAMES = 7



def index_tiff_frames(folder, pattern):
    files = {}
    for path in Path(folder).glob(pattern):
        match = re.search(r"T(\d+)\.tif$", path.name)
        if match:
            t = int(match.group(1))
            if t in files:
                raise ValueError(f"Duplicate T={t}: {path}")
            files[t] = path
    return dict(sorted(files.items()))


def check_odd_positive(n, name):
    if not isinstance(n, (int, np.integer)) or n < 1 or n % 2 != 1:
        raise ValueError(f"{name} must be a positive odd integer")


@dataclass
class VisualizationSettings:
    frame_interval_sec: float = 1.087
    fret_rolling_n: int = _FRET_AVERAGING_FRAMES
    ch1_rolling_n: int = _DONOR_AVERAGING_FRAMES
    baseline_n: int = 10
    min_overlap_frac: float = 0.05
    min_matched_frames: int = 1
    shrink_pixels: float = 0.8
    smooth_sigma: float = 1.8
    smooth_threshold: float = 0.5
    outline_color: str = "white"
    outline_width: float = 0.7
    ch1_vmin: float = 15
    ch1_vmax: float = 55
    ch3_vmin: float = 0
    ch3_vmax: float = 150
    fret_vmin: float = FRET_SCALE_MIN
    fret_vmax: float = FRET_SCALE_MAX
    error_type: str = "std"
    error_alpha: float = 0.20
    curve_color: str = "red"
    curve_linewidth: float = 2.5
    marker_color: str = MOVING_DOT_COLOR
    marker_size: float = 12
    zero_line_color: str = "white"
    x_lim: tuple | None = None
    y_lim: tuple | None = None
    combined_figsize: tuple = (12, 8)
    top_row_height: float = 1.0
    bottom_row_height: float = 1.0
    panel_gap: float = 0.10
    row_gap: float = 0.25
    colorbar_width: float = 0.035
    curve_width_factor: float = 0.65
    curve_height_factor: float = 1.0
    title_fontsize: float = 18
    axis_fontsize: float = 17
    tick_fontsize: float = 14
    time_fontsize: float = 20
    axis_linewidth: float = 1.5
    image_border_width: float = 1.5
    show_image_borders: bool = True
    fig_bg_color: str = "black"
    text_color: str = "white"
    fret_colormap: str = FRET_COLORMAP
    fret_colorbar_ticks: tuple | None = FRET_COLORBAR_TICKS


def make_fret_cmap(colormap_name=FRET_COLORMAP):
    """Preserve original piecewise coolwarm; allow selecting other Matplotlib maps."""
    base_cmap = plt.get_cmap(colormap_name)
    return LinearSegmentedColormap.from_list(f"display_{colormap_name}", [
        (0.00, base_cmap(0.00)),
        (0.05, base_cmap(0.00)),
        (0.50, base_cmap(0.50)),
        (0.95, base_cmap(1.00)),
        (1.00, base_cmap(1.00)),
    ], N=256)


class FRETVisualization:
    """Read prepared cropped TIFFs and precomputed FRET to render figures."""

    def __init__(self, data_dir, sample_name="mouse6", settings=None):
        self.data_dir = Path(data_dir)
        self.sample_name = sample_name
        self.s = settings if settings is not None else VisualizationSettings()
        s = self.s
        check_odd_positive(s.fret_rolling_n, "fret_rolling_n")
        check_odd_positive(s.ch1_rolling_n, "ch1_rolling_n")
        if s.error_type not in ("sem", "std"):
            raise ValueError("error_type must be 'sem' or 'std'")

        csv_path = self.data_dir / "cell_FRET.csv"
        self.df_fret = pd.read_csv(csv_path)
        required_columns = {"T", "cell_id", "mean_FRET"}
        if not required_columns.issubset(self.df_fret.columns):
            raise ValueError(f"Missing FRET CSV columns: {required_columns - set(self.df_fret.columns)}")
        self.df_fret["T"] = self.df_fret["T"].astype(int)

        name = self.sample_name
        self.mask_files = index_tiff_frames(self.data_dir / "cell_masks", f"{name}_labels_T*.tif")
        self.ch1_files = index_tiff_frames(self.data_dir / "donor", f"{name}_C001T*.tif")
        self.ch3_files = index_tiff_frames(self.data_dir / "acceptor", f"{name}_C002T*.tif")
        self.available_T = sorted(set(self.df_fret["T"]) & set(self.mask_files)
                                  & set(self.ch1_files) & set(self.ch3_files))
        if len(self.available_T) < s.fret_rolling_n:
            raise ValueError(f"Only {len(self.available_T)} matching frames")
        missing = set(self.df_fret["T"]) - set(self.available_T)
        if missing:
            raise ValueError(f"Missing image or mask files for frames: {sorted(missing)}")
        first_mask = tif.imread(self.mask_files[self.available_T[0]])
        self.crop_size = int(first_mask.shape[0])
        if first_mask.shape != (self.crop_size, self.crop_size):
            raise ValueError(f"Expected square masks; got {first_mask.shape}")

        half = s.fret_rolling_n // 2
        self.rolling_windows = []
        for i in range(half, len(self.available_T) - half):
            self.rolling_windows.append({
                "video_idx": len(self.rolling_windows),
                "window_T": self.available_T[i-half:i+half+1],
                "center_T": self.available_T[i],
                "center_index": i,
            })
        self.t0 = self.rolling_windows[0]["center_T"]
        for w in self.rolling_windows:
            w["time_sec"] = (w["center_T"] - self.t0) * s.frame_interval_sec

        self.fret_lookup = {
            int(t): {int(r.cell_id): float(r.mean_FRET)
                     for r in group.itertuples(index=False) if np.isfinite(r.mean_FRET)}
            for t, group in self.df_fret.groupby("T")
        }
        self.df_delta = self._build_delta_curve()
        self.ch1_cmap = LinearSegmentedColormap.from_list("black_cyan", ["black", "cyan"])
        self.ch3_cmap = LinearSegmentedColormap.from_list("black_red", ["black", "red"])
        self.fret_cmap = make_fret_cmap(s.fret_colormap)

    @lru_cache(maxsize=128)
    def load_cell_labels(self, t):
        labels = tif.imread(self.mask_files[int(t)])
        if labels.shape != (self.crop_size, self.crop_size):
            raise ValueError(f"T={t}: unexpected mask shape {labels.shape}")
        return labels

    def match_cells_by_overlap(self, ref, target):
        """Original.4 reciprocal best-overlap matching."""
        ref_ids = np.unique(ref[ref > 0])
        tgt_ids = np.unique(target[target > 0])
        if not len(ref_ids) or not len(tgt_ids):
            return {}
        overlap = np.array([
            [np.count_nonzero((ref == a) & (target == b)) for b in tgt_ids]
            for a in ref_ids
        ])
        best_tgt = overlap.argmax(axis=1)
        best_ref = overlap.argmax(axis=0)
        matches = {}
        for i, ref_id in enumerate(ref_ids):
            j = best_tgt[i]
            if best_ref[j] != i:
                continue
            intersection = overlap[i, j]
            ref_area = np.count_nonzero(ref == ref_id)
            tgt_area = np.count_nonzero(target == tgt_ids[j])
            if intersection / max(ref_area, tgt_area) >= self.s.min_overlap_frac:
                matches[int(ref_id)] = int(tgt_ids[j])
        return matches

    def refine_display_mask(self, mask):
        """Display morphology, applied to display only, not FRET values."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return mask.copy()
        smoothed = gaussian_filter(mask.astype(np.float32), sigma=self.s.smooth_sigma)
        refined = (smoothed >= self.s.smooth_threshold) & mask
        if self.s.shrink_pixels > 0:
            refined &= distance_transform_edt(refined) > self.s.shrink_pixels
        return refined

    def rolling_cell_fret(self, video_idx):
        """Original.4 per-cell matching/averaging."""
        window = self.rolling_windows[int(video_idx)]
        center_T = window["center_T"]
        center_labels = self.load_cell_labels(center_T)
        center_ids = np.unique(center_labels[center_labels > 0])
        values_by_cell = {int(cid): [] for cid in center_ids}
        for t in window["window_T"]:
            labels_t = self.load_cell_labels(t)
            matches = ({int(cid): int(cid) for cid in center_ids}
                       if t == center_T else self.match_cells_by_overlap(center_labels, labels_t))
            for center_id, target_id in matches.items():
                value = self.fret_lookup.get(int(t), {}).get(target_id, np.nan)
                if np.isfinite(value):
                    values_by_cell[center_id].append(value)
        display_map = np.full(center_labels.shape, np.nan, dtype=np.float32)
        display_labels = np.zeros(center_labels.shape, dtype=np.int32)
        rows = []
        for cid in center_ids:
            cid = int(cid)
            values = values_by_cell[cid]
            mean_fret = float(np.mean(values)) if len(values) >= self.s.min_matched_frames else np.nan
            if np.isfinite(mean_fret):
                refined_mask = self.refine_display_mask(center_labels == cid)
                if refined_mask.any():
                    display_map[refined_mask] = mean_fret
                    display_labels[refined_mask] = cid
            rows.append({"video_idx": int(video_idx), "center_T": center_T,
                         "cell_id": cid, "mean_FRET_rolling": mean_fret,
                         "n_matched_frames": len(values)})
        return display_map, display_labels, pd.DataFrame(rows)

    def get_ch1_image(self, video_idx):
        """Independent centered donor MIP; use available neighbors at edges."""
        center_idx = self.rolling_windows[int(video_idx)]["center_index"]
        half_ch1 = self.s.ch1_rolling_n // 2
        neighbors = self.available_T[max(0, center_idx-half_ch1):
                                     min(len(self.available_T), center_idx+half_ch1+1)]
        stack = [tif.imread(self.ch1_files[t]) for t in neighbors]
        return np.max(np.stack(stack), axis=0)

    def get_ch3_image(self, video_idx):
        """Raw acceptor image at center T, as in the original reference visualization."""
        center_T = self.rolling_windows[int(video_idx)]["center_T"]
        return tif.imread(self.ch3_files[center_T])

    def _build_delta_curve(self):
        s = self.s
        if not 1 <= s.baseline_n <= len(self.rolling_windows):
            raise ValueError("baseline_n must fit within available rolling windows")
        records = []
        for w in self.rolling_windows:
            i = w["video_idx"]
            _, _, summary = self.rolling_cell_fret(i)
            vals = summary["mean_FRET_rolling"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            mean = float(np.mean(vals)) if n else np.nan
            std = float(np.std(vals, ddof=1)) if n >= 2 else np.nan
            sem = std / np.sqrt(n) if n >= 2 else np.nan
            records.append({"video_idx": i, "center_T": w["center_T"],
                            "time_sec": w["time_sec"], "mean_FRET": mean,
                            "std_FRET": std, "sem_FRET": sem, "n_cells": n})
        df_delta = pd.DataFrame(records)
        self.fret_baseline = float(df_delta["mean_FRET"].iloc[:s.baseline_n].mean())
        if not np.isfinite(self.fret_baseline):
            raise ValueError("Cannot compute baseline from initial windows")
        df_delta["delta_FRET"] = df_delta["mean_FRET"] - self.fret_baseline
        df_delta["error"] = df_delta[f"{s.error_type}_FRET"]
        return df_delta

    def save_curve_csv(self, path):
        """Save only time, mouse-level cell-mean FRET, and cell-level error."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = self.df_delta[["time_sec", "mean_FRET", "error"]].copy()
        out = out.rename(columns={"error": "error_FRET"})
        out.to_csv(path, index=False)
        return path

    def plot_combined_frame(self, video_idx, show=False):
        """Original figure layout; returns a Matplotlib Figure for rendering/export."""
        s = self.s
        video_idx = int(video_idx)
        if not 0 <= video_idx < len(self.rolling_windows):
            raise IndexError(f"Invalid video_idx={video_idx}")
        w = self.rolling_windows[video_idx]
        time_sec = float(self.df_delta.iloc[video_idx]["time_sec"])
        donor = self.get_ch1_image(video_idx)
        acceptor = self.get_ch3_image(video_idx)
        fret_map, fret_labels, _ = self.rolling_cell_fret(video_idx)

        fig = plt.figure(figsize=s.combined_figsize, facecolor=s.fig_bg_color)
        gs = GridSpec(2, 4, figure=fig,
                      width_ratios=[1, 1, 1, s.colorbar_width],
                      height_ratios=[s.top_row_height, s.bottom_row_height],
                      wspace=s.panel_gap, hspace=s.row_gap)
        ax_ch1, ax_ch3, ax_fret = [fig.add_subplot(gs[0, k]) for k in range(3)]
        ax_cb = fig.add_subplot(gs[0, 3])
        ax_curve = fig.add_subplot(gs[1, :])
        for ax in (ax_ch1, ax_ch3, ax_fret, ax_cb, ax_curve):
            ax.set_facecolor("black")

        ax_ch1.imshow(donor, cmap=self.ch1_cmap, vmin=s.ch1_vmin,
                      vmax=s.ch1_vmax, interpolation="nearest")
        ax_ch3.imshow(acceptor, cmap=self.ch3_cmap, vmin=s.ch3_vmin,
                      vmax=s.ch3_vmax, interpolation="nearest")
        cmap = self.fret_cmap.copy()
        cmap.set_bad("black")
        norm = Normalize(vmin=s.fret_vmin, vmax=s.fret_vmax)
        ax_fret.imshow(np.ma.masked_invalid(fret_map), cmap=cmap,
                       norm=norm, interpolation="nearest")
        for cid in np.unique(fret_labels[fret_labels > 0]):
            ax_fret.contour((fret_labels == cid).astype(float), levels=[0.5],
                            colors=s.outline_color, linewidths=s.outline_width)

        for ax, title in zip((ax_ch1, ax_ch3, ax_fret),
                             ("Donor", "Acceptor", "FRET pseudocolor")):
            ax.set_title(title, fontsize=s.title_fontsize, color=s.text_color, pad=14)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(s.show_image_borders)
                spine.set_edgecolor("white")
                spine.set_linewidth(s.image_border_width)

        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax_cb,
                            ticks=s.fret_colorbar_ticks)
        cbar.set_label("FRET", fontsize=s.axis_fontsize, color=s.text_color, labelpad=12)
        cbar.ax.tick_params(colors="white", labelsize=s.tick_fontsize,
                            width=s.axis_linewidth, length=4)
        cbar.outline.set_edgecolor("white")
        cbar.outline.set_linewidth(1)

        x = self.df_delta["time_sec"].to_numpy(float)
        y = self.df_delta["delta_FRET"].to_numpy(float)
        err = self.df_delta["error"].to_numpy(float)
        ax_curve.fill_between(x, y-err, y+err, color=s.curve_color,
                              alpha=s.error_alpha, linewidth=0)
        ax_curve.plot(x, y, color=s.curve_color, linewidth=s.curve_linewidth)
        ax_curve.axhline(0, color=s.zero_line_color, linestyle="--", alpha=0.5)
        current_y = float(self.df_delta.iloc[video_idx]["delta_FRET"])
        if np.isfinite(current_y):
            ax_curve.plot(time_sec, current_y, "o", color=s.marker_color,
                          markersize=s.marker_size, markeredgecolor="white", zorder=10)
        ax_curve.axvline(time_sec, color=s.marker_color, linestyle="--", alpha=0.7)
        ax_curve.set_xlabel("Time (s)", fontsize=s.axis_fontsize,
                            color=s.text_color, labelpad=10)
        ax_curve.set_ylabel("ΔFRET", fontsize=s.axis_fontsize,
                            color=s.text_color, labelpad=10)
        ax_curve.set_xlim(s.x_lim if s.x_lim is not None else (0, max(float(np.nanmax(x)), 1.0)))
        if s.y_lim is not None:
            ax_curve.set_ylim(s.y_lim)
        ax_curve.tick_params(axis="both", colors="white", labelsize=s.tick_fontsize,
                             width=s.axis_linewidth, length=5)
        for spine in ax_curve.spines.values():
            spine.set_edgecolor("white")
            spine.set_linewidth(s.axis_linewidth)
        ax_curve.spines["top"].set_visible(False)
        ax_curve.spines["right"].set_visible(False)

        fig.suptitle(f"Time: {time_sec:.2f} s",
                     fontsize=s.time_fontsize, color=s.text_color, y=0.98)
        fig.subplots_adjust(left=0.08, right=0.94, top=0.88, bottom=0.10)
        pos = ax_curve.get_position()
        width = pos.width * s.curve_width_factor
        height = pos.height * s.curve_height_factor
        ax_curve.set_position([pos.x0 + (pos.width-width)/2,
                               pos.y0 + (pos.height-height)/2, width, height])
        if show:
            from IPython.display import display
            display(fig)
            plt.close(fig)
        return fig


def show_interactive(analysis, start_idx=0, interval_ms=120):
    """Original notebook Play + Slider behavior."""
    import ipywidgets as widgets
    from IPython.display import display
    n = len(analysis.rolling_windows)
    play = widgets.Play(value=start_idx, min=0, max=n-1, step=1,
                        interval=interval_ms)
    slider = widgets.IntSlider(value=start_idx, min=0, max=n-1, step=1,
                               description="Frame", continuous_update=False)
    widgets.jslink((play, "value"), (slider, "value"))
    output = widgets.interactive_output(
        lambda video_idx: analysis.plot_combined_frame(video_idx, show=True),
        {"video_idx": slider}
    )
    display(widgets.VBox([widgets.HBox([play, slider]), output]))
    return play, slider, output


def export_video(analysis, output_path, indices=None):
    """Render visualization frames to an MP4 or AVI; requires imageio and FFmpeg backend.

    The data/averaging are unchanged. This is a new output feature not present
    in the reference notebook. Frame spacing uses the chosen constant fps;
    measurement times shown in the figure follow original T timestamps.
    """
    import imageio.v2 as imageio
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if indices is None:
        indices = range(len(analysis.rolling_windows))
    # ImageIO ffmpeg plugin handles mp4/mov/avi according to filename.
    with imageio.get_writer(
    str(path),
    format="FFMPEG",
    mode="I",
    fps=VIDEO_FPS,
    codec="libx264",
    macro_block_size=1) as writer:
        for idx in indices:
            fig = analysis.plot_combined_frame(int(idx), show=False)
            fig.set_dpi(VIDEO_DPI)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            frame = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(frame)
            plt.close(fig)
    return path
