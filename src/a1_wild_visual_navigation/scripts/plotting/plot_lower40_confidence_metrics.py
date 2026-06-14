#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_VELOCITY_FRAMES = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames"
)
DEFAULT_FORCE_FRAMES = Path(
    "/home/rain/github_upload/Result/force_default_normalize/wvn_frames"
)
DEFAULT_OUTPUT_DIR = Path("/home/rain/github_upload/Result/rendered_figures")
DEFAULT_FRAMES = "420,700,1000,1520,2040,2420,2700,2820,3340"
DEFAULT_RANGE_START = 420
DEFAULT_RANGE_END = 3340


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot lower-40% confidence metrics from saved confidence.npy maps."
    )
    parser.add_argument("--velocity-frames-dir", type=Path, default=DEFAULT_VELOCITY_FRAMES)
    parser.add_argument("--force-frames-dir", type=Path, default=DEFAULT_FORCE_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument("--range-start", type=int, default=DEFAULT_RANGE_START)
    parser.add_argument("--range-end", type=int, default=DEFAULT_RANGE_END)
    parser.add_argument("--low-threshold", type=float, default=0.2)
    return parser.parse_args()


def parse_frames(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_manifest_rows(frames_dir):
    manifest_path = frames_dir / "manifest.csv"
    with manifest_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def compute_lower40_rows(frames_dir, manifest_rows, range_start, range_end, low_threshold):
    rows = []
    for row in manifest_rows:
        frame_index = int(row["frame_index"])
        if frame_index < range_start or frame_index > range_end:
            continue

        confidence_path = frames_dir / row["frame_dir"] / "confidence.npy"
        if not confidence_path.exists():
            continue

        arr = np.load(confidence_path)
        if arr.ndim != 2:
            continue

        height = arr.shape[0]
        lower40 = arr[int(round(0.6 * height)) :, :]

        rows.append(
            {
                "frame_index": frame_index,
                "timestamp": float(row["stamp_secs"]) + float(row["stamp_nsecs"]) * 1e-9,
                "instant_traversability": float(row["instant_traversability"])
                if row.get("instant_traversability")
                else np.nan,
                "lower40_mean": float(lower40.mean()),
                "lower40_std": float(lower40.std()),
                "lower40_low_ratio": float((lower40 < low_threshold).mean()),
                "lower40_low_ratio_01": float((lower40 < 0.1).mean()),
            }
        )
    return rows


def write_summary_csv(rows, output_path):
    fieldnames = [
        "frame_index",
        "timestamp",
        "instant_traversability",
        "lower40_mean",
        "lower40_std",
        "lower40_low_ratio",
        "lower40_low_ratio_01",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metrics(rows, representative_frames, title, output_path, low_threshold):
    if not rows:
        raise RuntimeError(f"No lower-40% confidence rows available for {title}")

    frame_indices = [row["frame_index"] for row in rows]
    mean_values = [row["lower40_mean"] for row in rows]
    low_ratio_values = [row["lower40_low_ratio"] for row in rows]

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
    top_ax.plot(frame_indices, mean_values, color="#1f77b4", linewidth=1.6, label="Lower 40% mean confidence")
    top_ax.set_title("Ground-Region Mean Confidence", fontsize=13)
    top_ax.set_ylabel("Mean confidence")
    top_ax.grid(True, which="major", alpha=0.4)
    top_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    top_ax.minorticks_on()
    top_ax.legend(loc="upper right", frameon=True)

    bottom_ax.plot(
        frame_indices,
        low_ratio_values,
        color="#d62728",
        linewidth=1.6,
        label=f"Lower 40% low-confidence ratio (< {low_threshold:.1f})",
    )
    bottom_ax.set_title("Ground-Region Low-Confidence Ratio", fontsize=13)
    bottom_ax.set_ylabel("Low-confidence ratio")
    bottom_ax.set_xlabel("Frame Index")
    bottom_ax.grid(True, which="major", alpha=0.4)
    bottom_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    bottom_ax.minorticks_on()
    bottom_ax.legend(loc="upper right", frameon=True)

    y_top = max(mean_values) if mean_values else 0.0
    for idx, frame in enumerate(representative_frames):
        if frame < frame_indices[0] or frame > frame_indices[-1]:
            continue
        top_ax.axvline(frame, color="#6c757d", linewidth=0.8, alpha=0.6)
        bottom_ax.axvline(frame, color="#6c757d", linewidth=0.8, alpha=0.6)
        text_y = y_top * 0.98 if y_top > 0 else 0.0
        offset = 0.0 if idx % 2 == 0 else -0.04 * y_top
        top_ax.text(
            frame,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    velocity_manifest_rows = load_manifest_rows(args.velocity_frames_dir)
    force_manifest_rows = load_manifest_rows(args.force_frames_dir)

    velocity_rows = compute_lower40_rows(
        frames_dir=args.velocity_frames_dir,
        manifest_rows=velocity_manifest_rows,
        range_start=args.range_start,
        range_end=args.range_end,
        low_threshold=args.low_threshold,
    )
    force_rows = compute_lower40_rows(
        frames_dir=args.force_frames_dir,
        manifest_rows=force_manifest_rows,
        range_start=args.range_start,
        range_end=args.range_end,
        low_threshold=args.low_threshold,
    )

    velocity_plot = args.output_dir / "lower40_confidence_velocity_0420_3340.png"
    force_plot = args.output_dir / "lower40_confidence_force_0420_3340.png"
    velocity_csv = args.output_dir / "lower40_confidence_velocity_0420_3340.csv"
    force_csv = args.output_dir / "lower40_confidence_force_0420_3340.csv"

    plot_metrics(
        rows=velocity_rows,
        representative_frames=representative_frames,
        title="Velocity Run: Lower 40% Confidence Metrics",
        output_path=velocity_plot,
        low_threshold=args.low_threshold,
    )
    plot_metrics(
        rows=force_rows,
        representative_frames=representative_frames,
        title="Force Run: Lower 40% Confidence Metrics",
        output_path=force_plot,
        low_threshold=args.low_threshold,
    )
    write_summary_csv(velocity_rows, velocity_csv)
    write_summary_csv(force_rows, force_csv)

    print(f"[ok] wrote {velocity_plot}")
    print(f"[ok] wrote {force_plot}")
    print(f"[ok] wrote {velocity_csv}")
    print(f"[ok] wrote {force_csv}")


if __name__ == "__main__":
    main()
