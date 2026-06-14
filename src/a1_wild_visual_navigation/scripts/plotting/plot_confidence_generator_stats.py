#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_VELOCITY_CSV = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_velocity_debug.csv"
)
DEFAULT_FORCE_CSV = Path(
    "/home/rain/github_upload/Result/force_default_normalize/wvn_force_debug.csv"
)
DEFAULT_VELOCITY_MANIFEST = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames/manifest.csv"
)
DEFAULT_FORCE_MANIFEST = Path(
    "/home/rain/github_upload/Result/force_default_normalize/wvn_frames/manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/rain/github_upload/Result/rendered_figures")
DEFAULT_FRAMES = "420,700,1000,1520,2040,2420,2700,2820,3340"
DEFAULT_RANGE_START = 420
DEFAULT_RANGE_END = 3340


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot confidence-generator mean/std over time and mark representative frames."
    )
    parser.add_argument("--velocity-csv", type=Path, default=DEFAULT_VELOCITY_CSV)
    parser.add_argument("--force-csv", type=Path, default=DEFAULT_FORCE_CSV)
    parser.add_argument("--velocity-manifest", type=Path, default=DEFAULT_VELOCITY_MANIFEST)
    parser.add_argument("--force-manifest", type=Path, default=DEFAULT_FORCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--range-start", type=int, default=DEFAULT_RANGE_START)
    parser.add_argument("--range-end", type=int, default=DEFAULT_RANGE_END)
    return parser.parse_args()


def parse_frames(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def confidence_series(rows):
    timestamps = []
    mean_values = []
    std_values = []
    for row in rows:
        mean = row.get("confidence_generator_mean", "")
        std = row.get("confidence_generator_std", "")
        if mean in ("", None) or std in ("", None):
            continue
        timestamps.append(float(row["timestamp"]))
        mean_values.append(float(mean))
        std_values.append(float(std))
    return timestamps, mean_values, std_values


def representative_frame_times(manifest_rows, representative_frames):
    frame_to_time = {}
    for row in manifest_rows:
        frame_index = int(row["frame_index"])
        if frame_index in representative_frames:
            frame_to_time[frame_index] = float(row["stamp_secs"]) + float(row["stamp_nsecs"]) * 1e-9
    return frame_to_time


def manifest_frame_time(manifest_rows, target_frame):
    for row in manifest_rows:
        if int(row["frame_index"]) == target_frame:
            return float(row["stamp_secs"]) + float(row["stamp_nsecs"]) * 1e-9
    raise KeyError(f"Frame {target_frame} not found in manifest")


def plot_run(csv_rows, manifest_rows, representative_frames, range_start, range_end, title, output_path):
    timestamps, mean_values, std_values = confidence_series(csv_rows)
    if not timestamps:
        raise RuntimeError(f"No confidence-generator mean/std samples found for {title}")

    start_time = manifest_frame_time(manifest_rows, range_start)
    end_time = manifest_frame_time(manifest_rows, range_end)

    cropped = [
        (t, m, s)
        for t, m, s in zip(timestamps, mean_values, std_values)
        if start_time <= t <= end_time
    ]
    if not cropped:
        raise RuntimeError(f"No confidence-generator samples inside frame window [{range_start}, {range_end}]")

    t0 = start_time
    rel_times = [t - t0 for t, _, _ in cropped]
    mean_values = [m for _, m, _ in cropped]
    std_values = [s for _, _, s in cropped]
    frame_times = representative_frame_times(manifest_rows, representative_frames)
    rel_frame_times = {frame: ts - t0 for frame, ts in frame_times.items() if start_time <= ts <= end_time}

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8.5),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    fig.suptitle(title, fontsize=18)

    top_ax, bottom_ax = axes
    top_ax.plot(rel_times, mean_values, color="#1f77b4", linewidth=1.6, label="Confidence mean")
    top_ax.set_title("Confidence Generator Mean", fontsize=13)
    top_ax.set_ylabel("Mean")
    top_ax.grid(True, which="major", alpha=0.4)
    top_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    top_ax.minorticks_on()
    top_ax.legend(loc="upper right", frameon=True)

    bottom_ax.plot(rel_times, std_values, color="#ff7f0e", linewidth=1.6, label="Confidence std")
    bottom_ax.set_title("Confidence Generator Std", fontsize=13)
    bottom_ax.set_ylabel("Std")
    bottom_ax.set_xlabel("Time [s]")
    bottom_ax.grid(True, which="major", alpha=0.4)
    bottom_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    bottom_ax.minorticks_on()
    bottom_ax.legend(loc="upper right", frameon=True)

    y_top = max(mean_values) if mean_values else 0.0
    for idx, frame in enumerate(representative_frames):
        if frame not in rel_frame_times:
            continue
        x = rel_frame_times[frame]
        top_ax.axvline(x, color="#6c757d", linewidth=0.8, alpha=0.6)
        bottom_ax.axvline(x, color="#6c757d", linewidth=0.8, alpha=0.6)
        text_y = y_top * 0.98 if y_top > 0 else 0.0
        offset = 0.0 if idx % 2 == 0 else -0.04 * y_top
        top_ax.text(
            x,
            text_y + offset,
            str(frame),
            rotation=90,
            va="top",
            ha="center",
            fontsize=8,
            color="#495057",
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 0.5},
        )

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    representative_frames = parse_frames(args.frames)

    velocity_csv_rows = load_csv_rows(args.velocity_csv)
    force_csv_rows = load_csv_rows(args.force_csv)
    velocity_manifest_rows = load_manifest_rows(args.velocity_manifest)
    force_manifest_rows = load_manifest_rows(args.force_manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    velocity_output = args.output_dir / "confidence_generator_velocity_over_time.png"
    force_output = args.output_dir / "confidence_generator_force_over_time.png"

    plot_run(
        csv_rows=velocity_csv_rows,
        manifest_rows=velocity_manifest_rows,
        representative_frames=representative_frames,
        range_start=args.range_start,
        range_end=args.range_end,
        title="Velocity Run: Confidence Generator Mean and Std Over Time",
        output_path=velocity_output,
    )
    plot_run(
        csv_rows=force_csv_rows,
        manifest_rows=force_manifest_rows,
        representative_frames=representative_frames,
        range_start=args.range_start,
        range_end=args.range_end,
        title="Force Run: Confidence Generator Mean and Std Over Time",
        output_path=force_output,
    )

    print(f"[ok] wrote {velocity_output}")
    print(f"[ok] wrote {force_output}")


if __name__ == "__main__":
    main()
