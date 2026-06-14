#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the clearest terrain-dependent aggregate force metrics."
    )
    parser.add_argument(
        "--summary-csv",
        default="/home/rain/github_upload/Result/rendered_figures/aggregate_force_pattern_summary.csv",
        help="Aggregate force summary CSV.",
    )
    parser.add_argument(
        "--output",
        default="/home/rain/github_upload/Result/rendered_figures/force_variability_by_terrain.png",
        help="Output path for the legacy combined figure.",
    )
    parser.add_argument(
        "--std-output",
        default="/home/rain/github_upload/Result/rendered_figures/force_error_std_by_terrain.png",
        help="Output path for the force variability figure.",
    )
    parser.add_argument(
        "--peak-output",
        default="/home/rain/github_upload/Result/rendered_figures/force_peak_rate_by_terrain.png",
        help="Output path for the peak-density figure.",
    )
    return parser.parse_args()


def load_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def short_label(label: str) -> str:
    mapping = {
        "Carpeted corridor": "Carpet corridor",
        "Tile floor": "Tile floor",
        "Brick pavement": "Brick pavement",
        "Brick pavement to grass transition": "Brick->grass",
        "Grass": "Grass",
        "Slabs pavement": "Slabs",
        "Gravel": "Gravel",
        "Brick pavement and grass mixture": "Brick+grass mix",
    }
    return mapping.get(label, label)


def style_axes(ax, x, labels):
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(True, axis="x", alpha=0.1)


def plot_force_std(labels, x, force_std, output_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Force Variability by Terrain Window", fontsize=22, fontweight="bold")

    bars = ax.bar(x, force_std, color="#E67E22", alpha=0.92)
    ax.set_title("Standard Deviation of Filtered Force Error", fontsize=17)
    ax.set_ylabel("Force error std", fontsize=13)
    style_axes(ax, x, labels)

    y_pad = max(force_std.max() * 0.04, 0.001)
    ax.set_ylim(0.0, force_std.max() + y_pad * 3.0)
    for bar, value in zip(bars, force_std):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + y_pad,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add a low-range zoom so the stable windows do not visually collapse into
    # the same rounded value in the main axis.
    low_mask = force_std <= 0.012
    if np.any(low_mask):
        inset = ax.inset_axes([0.52, 0.42, 0.42, 0.46])
        low_x = x[low_mask]
        low_vals = force_std[low_mask]
        low_labels = [labels[i] for i, keep in enumerate(low_mask) if keep]
        inset_bars = inset.bar(np.arange(len(low_vals)), low_vals, color="#E67E22", alpha=0.92)
        inset.set_title("Low-variability zoom", fontsize=10)
        inset.set_ylabel("Std", fontsize=9)
        inset.set_xticks(np.arange(len(low_vals)))
        inset.set_xticklabels(low_labels, rotation=25, ha="right", fontsize=7)
        inset.tick_params(axis="y", labelsize=8)
        inset.grid(True, axis="y", alpha=0.3)
        low_min = max(0.0, low_vals.min() - 0.0007)
        low_max = low_vals.max() + 0.0009
        inset.set_ylim(low_min, low_max)
        low_pad = max((low_max - low_min) * 0.06, 0.00008)
        for bar, value in zip(inset_bars, low_vals):
            inset.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + low_pad,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_peak_rate(labels, x, peak_rate, output_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Force Peak Density by Terrain Window", fontsize=22, fontweight="bold")

    bars = ax.bar(x, peak_rate, color="#2CA02C", alpha=0.92)
    ax.set_title("Peak Rate of Filtered Force Error", fontsize=17)
    ax.set_ylabel("Peak rate (Hz)", fontsize=13)
    style_axes(ax, x, labels)

    y_pad = max(peak_rate.max() * 0.04, 0.02)
    ax.set_ylim(0.0, peak_rate.max() + y_pad * 3.0 if peak_rate.max() > 0 else 1.0)
    for bar, value in zip(bars, peak_rate):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + y_pad,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_combined(labels, x, force_std, peak_rate, output_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Force Variability by Terrain Window", fontsize=24, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(x, force_std, color="#E67E22", alpha=0.9)
    ax.set_title("Standard Deviation of Filtered Force Error", fontsize=17)
    ax.set_ylabel("Force error std", fontsize=13)
    style_axes(ax, x, labels)
    for bar, value in zip(bars, force_std):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax = axes[1]
    bars = ax.bar(x, peak_rate, color="#2CA02C", alpha=0.9)
    ax.set_title("Peak Rate of Filtered Force Error", fontsize=17)
    ax.set_ylabel("Peak rate (Hz)", fontsize=13)
    style_axes(ax, x, labels)
    for bar, value in zip(bars, peak_rate):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(Path(args.summary_csv))

    labels = [f"{row['frame_start']}-{row['frame_end']}\n{short_label(row['terrain_label'])}" for row in rows]
    x = np.arange(len(rows))

    force_std = np.array([float(row["force_error_std"]) for row in rows], dtype=float)
    peak_rate = np.array([float(row["peak_rate_hz"]) for row in rows], dtype=float)

    combined_output = Path(args.output)
    std_output = Path(args.std_output)
    peak_output = Path(args.peak_output)

    plot_combined(labels, x, force_std, peak_rate, combined_output)
    plot_force_std(labels, x, force_std, std_output)
    plot_peak_rate(labels, x, peak_rate, peak_output)

    print(f"[ok] wrote {combined_output}")
    print(f"[ok] wrote {std_output}")
    print(f"[ok] wrote {peak_output}")


if __name__ == "__main__":
    main()
