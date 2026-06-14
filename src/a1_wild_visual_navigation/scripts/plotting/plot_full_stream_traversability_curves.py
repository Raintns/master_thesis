#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_VELOCITY_MANIFEST = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames/manifest.csv"
)
DEFAULT_FORCE_MANIFEST = Path(
    "/home/rain/github_upload/Result/force_default_normalize/wvn_frames/manifest.csv"
)
DEFAULT_VELOCITY_CSV = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_velocity_debug.csv"
)
DEFAULT_FORCE_CSV = Path(
    "/home/rain/github_upload/Result/force_default_normalize/wvn_force_debug.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/rain/github_upload/Result/rendered_figures")
DEFAULT_RANGE_START = 420
DEFAULT_RANGE_END = 3340
DEFAULT_REPRESENTATIVE_FRAMES = "420,700,1000,1520,2040,2420,2700,2820,3340"
TERRAIN_WINDOWS = [
    (420, 770, "Carpet"),
    (770, 1000, "Tile"),
    (1000, 2140, "Brick"),
    (2140, 2780, "Grass"),
    (2780, 2820, "Slabs"),
    (2820, 2900, "Gravel"),
    (2900, 3080, "Brick+Grass"),
    (3080, 3340, "Brick"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot full WVN debug-stream traversability and error curves over an interpolated frame-index axis."
        )
    )
    parser.add_argument("--velocity-manifest", type=Path, default=DEFAULT_VELOCITY_MANIFEST)
    parser.add_argument("--force-manifest", type=Path, default=DEFAULT_FORCE_MANIFEST)
    parser.add_argument("--velocity-csv", type=Path, default=DEFAULT_VELOCITY_CSV)
    parser.add_argument("--force-csv", type=Path, default=DEFAULT_FORCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--range-start", type=int, default=DEFAULT_RANGE_START)
    parser.add_argument("--range-end", type=int, default=DEFAULT_RANGE_END)
    parser.add_argument("--representative-frames", default=DEFAULT_REPRESENTATIVE_FRAMES)
    parser.add_argument(
        "--tau-window",
        type=int,
        default=101,
        help="Centered rolling window for the traversability trend line.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to the output filenames, e.g. '_50samples'.",
    )
    return parser.parse_args()


def parse_frames(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def load_manifest(path):
    frame = pd.read_csv(path)
    frame["timestamp"] = frame["stamp_secs"] + frame["stamp_nsecs"] * 1e-9
    return frame.sort_values("frame_index").reset_index(drop=True)


def frame_range_timestamps(manifest, frame_start, frame_end):
    timestamps = np.interp(
        [frame_start, frame_end],
        manifest["frame_index"].to_numpy(dtype=float),
        manifest["timestamp"].to_numpy(dtype=float),
    )
    return float(timestamps[0]), float(timestamps[1])


def load_debug_window(path, time_start, time_end):
    frame = pd.read_csv(path)
    frame = frame[(frame["timestamp"] >= time_start) & (frame["timestamp"] <= time_end)].copy()
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def add_interpolated_frame_index(debug_frame, manifest):
    debug_frame["frame_index_interp"] = np.interp(
        debug_frame["timestamp"].to_numpy(dtype=float),
        manifest["timestamp"].to_numpy(dtype=float),
        manifest["frame_index"].to_numpy(dtype=float),
    )
    return debug_frame


def series(frame, x_col, y_col):
    subset = frame[[x_col, y_col]].dropna()
    return subset[x_col].to_numpy(dtype=float), subset[y_col].to_numpy(dtype=float)


def rolling_mean(values, window):
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def representative_guides(ax, frames):
    for frame_index in frames:
        ax.axvline(frame_index, color="#d0d0d0", linewidth=0.7, alpha=0.7, zorder=0)


def add_window_shading(ax, frame_start, frame_end):
    colors = ("#f6f8fb", "#eef3f9")
    for index, (start, end, _label) in enumerate(TERRAIN_WINDOWS):
        left = max(frame_start, start)
        right = min(frame_end, end)
        if right <= left:
            continue
        ax.axvspan(left, right, color=colors[index % 2], alpha=0.55, zorder=0)


def plot_full_curve(
    debug_frame,
    frame_start,
    frame_end,
    representative_frames,
    title,
    tau_col,
    error_raw_col,
    error_filtered_col,
    error_label,
    tau_window,
):
    tau_x, tau_y = series(debug_frame, "frame_index_interp", tau_col)
    raw_x, raw_y = series(debug_frame, "frame_index_interp", error_raw_col)
    filtered_x, filtered_y = series(debug_frame, "frame_index_interp", error_filtered_col)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.08]},
    )
    fig.suptitle(title, fontsize=18, fontweight="normal")

    top_ax, bottom_ax = axes

    representative_guides(top_ax, representative_frames)
    representative_guides(bottom_ax, representative_frames)

    top_ax.plot(
        tau_x,
        tau_y,
        color="#8fbfe8",
        linewidth=0.9,
        alpha=0.55,
        label=f"{tau_col.replace('_', ' ').title()} (raw)",
    )
    top_ax.plot(
        tau_x,
        rolling_mean(tau_y, tau_window),
        color="#1f77b4",
        linewidth=1.9,
        label=f"{tau_col.replace('_', ' ').title()} (trend)",
    )
    top_ax.set_title(tau_col.replace("_", " ").title(), fontsize=14)
    top_ax.set_ylabel("Traversability score τ")
    top_ax.grid(True, which="major", alpha=0.45)
    top_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    top_ax.minorticks_on()
    top_ax.legend(loc="lower left", frameon=True)

    bottom_ax.plot(
        raw_x,
        raw_y,
        color="#ff7f0e",
        linewidth=1.2,
        alpha=0.9,
        label=f"{error_label} MSE",
    )
    bottom_ax.plot(
        filtered_x,
        filtered_y,
        color="#2ca02c",
        linewidth=1.7,
        linestyle="--",
        label=f"{error_label} filtered",
    )
    bottom_ax.set_title(error_label, fontsize=14)
    bottom_ax.set_xlabel("Frame Index (interpolated from debug timestamp)")
    bottom_ax.set_ylabel(error_label)
    bottom_ax.grid(True, which="major", alpha=0.45)
    bottom_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    bottom_ax.minorticks_on()
    bottom_ax.legend(loc="upper left", frameon=True)

    bottom_ax.set_xlim(frame_start, frame_end)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def plot_clean_curve(
    debug_frame,
    frame_start,
    frame_end,
    representative_frames,
    title,
    tau_col,
    error_filtered_col,
    error_label,
    tau_window,
):
    tau_x, tau_y = series(debug_frame, "frame_index_interp", tau_col)
    filtered_x, filtered_y = series(debug_frame, "frame_index_interp", error_filtered_col)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15.5, 7.2),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )
    fig.suptitle(title, fontsize=18, fontweight="normal")

    top_ax, bottom_ax = axes
    for ax in axes:
        add_window_shading(ax, frame_start, frame_end)
        representative_guides(ax, representative_frames)
        ax.grid(True, which="major", alpha=0.28)
        ax.grid(True, which="minor", alpha=0.12, linestyle=":")
        ax.minorticks_on()

    tau_trend = rolling_mean(tau_y, tau_window)
    filtered_trend = rolling_mean(filtered_y, max(11, tau_window // 3))

    top_ax.plot(
        tau_x,
        tau_trend,
        color="#1f77b4",
        linewidth=2.0,
        label="Supervision traversability",
    )
    top_ax.set_title("Traversability trend", fontsize=13)
    top_ax.set_ylabel("Traversability score τ")
    top_ax.legend(loc="lower left", frameon=True)

    bottom_ax.plot(
        filtered_x,
        filtered_y,
        color="#a8d5a2",
        linewidth=1.0,
        alpha=0.45,
        label=f"{error_label} filtered (raw)",
    )
    bottom_ax.plot(
        filtered_x,
        filtered_trend,
        color="#2ca02c",
        linewidth=2.0,
        linestyle="--",
        label=f"{error_label} filtered (trend)",
    )
    bottom_ax.set_title(f"{error_label} trend", fontsize=13)
    bottom_ax.set_xlabel("Frame Index")
    bottom_ax.set_ylabel(error_label)
    bottom_ax.legend(loc="upper left", frameon=True)
    bottom_ax.set_xlim(frame_start, frame_end)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def plot_combined_curve(
    debug_frame,
    frame_start,
    frame_end,
    representative_frames,
    title,
    tau_col,
    error_filtered_col,
    error_label,
    tau_window,
):
    tau_x, tau_y = series(debug_frame, "frame_index_interp", tau_col)
    filtered_x, filtered_y = series(debug_frame, "frame_index_interp", error_filtered_col)

    tau_trend = rolling_mean(tau_y, tau_window)
    filtered_trend = rolling_mean(filtered_y, max(11, tau_window // 3))

    trend_min = float(np.min(tau_trend))
    trend_max = float(np.max(tau_trend))
    trend_pad = max((trend_max - trend_min) * 0.12, 1e-5)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.5, 9.2),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.16},
    )
    fig.suptitle(title, fontsize=18, fontweight="normal")

    top_ax, mid_ax, bottom_ax = axes
    for ax in axes:
        add_window_shading(ax, frame_start, frame_end)
        representative_guides(ax, representative_frames)
        ax.grid(True, which="major", alpha=0.28)
        ax.grid(True, which="minor", alpha=0.12, linestyle=":")
        ax.minorticks_on()
        ax.set_xlim(frame_start, frame_end)

    top_ax.plot(
        tau_x,
        tau_y,
        color="#96a0ad",
        linewidth=1.05,
        alpha=0.8,
        label="Instant traversability (raw)",
    )
    top_ax.plot(
        tau_x,
        tau_trend,
        color="#1f77b4",
        linewidth=2.0,
        label="Instant traversability (trend)",
    )
    top_ax.set_title("Instant traversability", fontsize=13)
    top_ax.set_ylabel("Traversability score τ")
    top_ax.legend(loc="lower left", frameon=True)

    mid_ax.plot(
        tau_x,
        tau_y,
        color="#96a0ad",
        linewidth=0.95,
        alpha=0.72,
        label="Instant traversability (raw)",
    )
    mid_ax.plot(
        tau_x,
        tau_trend,
        color="#1f77b4",
        linewidth=2.1,
        label="Instant traversability (trend)",
    )
    mid_ax.set_title("Instant traversability (zoomed trend)", fontsize=13)
    mid_ax.set_ylabel("Traversability score τ")
    mid_ax.set_ylim(trend_min - trend_pad, trend_max + trend_pad)
    mid_ax.legend(loc="lower left", frameon=True)

    bottom_ax.plot(
        filtered_x,
        filtered_y,
        color="#f28e2b",
        linewidth=1.05,
        alpha=0.62,
        label=f"{error_label} filtered (raw)",
    )
    bottom_ax.plot(
        filtered_x,
        filtered_trend,
        color="#2ca02c",
        linewidth=2.0,
        linestyle="--",
        label=f"{error_label} filtered (trend)",
    )
    bottom_ax.set_title(f"{error_label} filtered", fontsize=13)
    bottom_ax.set_xlabel("Frame Index")
    bottom_ax.set_ylabel(error_label)
    bottom_ax.legend(loc="upper left", frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def main():
    args = parse_args()
    representative_frames = parse_frames(args.representative_frames)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.suffix

    velocity_manifest = load_manifest(args.velocity_manifest)
    force_manifest = load_manifest(args.force_manifest)

    vel_t0, vel_t1 = frame_range_timestamps(velocity_manifest, args.range_start, args.range_end)
    force_t0, force_t1 = frame_range_timestamps(force_manifest, args.range_start, args.range_end)

    velocity_debug = add_interpolated_frame_index(
        load_debug_window(args.velocity_csv, vel_t0, vel_t1),
        velocity_manifest,
    )
    force_debug = add_interpolated_frame_index(
        load_debug_window(args.force_csv, force_t0, force_t1),
        force_manifest,
    )

    velocity_fig = plot_full_curve(
        debug_frame=velocity_debug,
        frame_start=args.range_start,
        frame_end=args.range_end,
        representative_frames=representative_frames,
        title="Traversability and Velocity Error Over Time (Full Debug Stream)",
        tau_col="supervision_traversability",
        error_raw_col="velocity_tracking_error_mse",
        error_filtered_col="velocity_tracking_error_filtered",
        error_label="Velocity error",
        tau_window=args.tau_window,
    )
    force_fig = plot_full_curve(
        debug_frame=force_debug,
        frame_start=args.range_start,
        frame_end=args.range_end,
        representative_frames=representative_frames,
        title="Traversability and Force Error Over Time (Full Debug Stream)",
        tau_col="supervision_traversability",
        error_raw_col="force_error_mse",
        error_filtered_col="force_error_filtered",
        error_label="Force error",
        tau_window=args.tau_window,
    )
    velocity_clean_fig = plot_clean_curve(
        debug_frame=velocity_debug,
        frame_start=args.range_start,
        frame_end=args.range_end,
        representative_frames=representative_frames,
        title="Traversability and Velocity Error Over Time",
        tau_col="supervision_traversability",
        error_filtered_col="velocity_tracking_error_filtered",
        error_label="Velocity error",
        tau_window=args.tau_window,
    )
    force_clean_fig = plot_clean_curve(
        debug_frame=force_debug,
        frame_start=args.range_start,
        frame_end=args.range_end,
        representative_frames=representative_frames,
        title="Traversability and Force Error Over Time",
        tau_col="supervision_traversability",
        error_filtered_col="force_error_filtered",
        error_label="Force error",
        tau_window=args.tau_window,
    )
    velocity_combined_fig = plot_combined_curve(
        debug_frame=velocity_debug,
        frame_start=args.range_start,
        frame_end=args.range_end,
        representative_frames=representative_frames,
        title="Traversability and Velocity Error Over Time",
        tau_col="instant_traversability",
        error_filtered_col="velocity_tracking_error_filtered",
        error_label="Velocity error",
        tau_window=args.tau_window,
    )
    force_combined_fig = plot_combined_curve(
        debug_frame=force_debug,
        frame_start=args.range_start,
        frame_end=args.range_end,
        representative_frames=representative_frames,
        title="Traversability and Force Error Over Time",
        tau_col="instant_traversability",
        error_filtered_col="force_error_filtered",
        error_label="Force error",
        tau_window=args.tau_window,
    )

    velocity_path = args.output_dir / f"traversability_velocity_curves_0420_3340_fullstream{suffix}.png"
    force_path = args.output_dir / f"traversability_force_curves_0420_3340_fullstream{suffix}.png"
    velocity_clean_path = args.output_dir / f"traversability_velocity_curves_0420_3340_fullstream_clean{suffix}.png"
    force_clean_path = args.output_dir / f"traversability_force_curves_0420_3340_fullstream_clean{suffix}.png"
    velocity_combined_path = args.output_dir / f"traversability_velocity_curves_0420_3340_combined{suffix}.png"
    force_combined_path = args.output_dir / f"traversability_force_curves_0420_3340_combined{suffix}.png"
    velocity_fig.savefig(velocity_path, facecolor="white")
    force_fig.savefig(force_path, facecolor="white")
    velocity_clean_fig.savefig(velocity_clean_path, facecolor="white")
    force_clean_fig.savefig(force_clean_path, facecolor="white")
    velocity_combined_fig.savefig(velocity_combined_path, facecolor="white")
    force_combined_fig.savefig(force_combined_path, facecolor="white")
    plt.close(velocity_fig)
    plt.close(force_fig)
    plt.close(velocity_clean_fig)
    plt.close(force_clean_fig)
    plt.close(velocity_combined_fig)
    plt.close(force_combined_fig)

    print(f"[ok] wrote {velocity_path}")
    print(f"[ok] wrote {force_path}")
    print(f"[ok] wrote {velocity_clean_path}")
    print(f"[ok] wrote {force_clean_path}")
    print(f"[ok] wrote {velocity_combined_path}")
    print(f"[ok] wrote {force_combined_path}")


if __name__ == "__main__":
    main()
