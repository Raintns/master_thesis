#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot terrain-window summaries for velocity/force error and traversability."
    )
    parser.add_argument(
        "--summary-csv",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_signal_summary.csv",
        help="CSV generated from terrain-window analysis.",
    )
    parser.add_argument(
        "--plot-output",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_signal_panels.png",
        help="Path for the comparison plot image.",
    )
    parser.add_argument(
        "--table-output",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_signal_table.png",
        help="Path for the rendered summary table image.",
    )
    return parser.parse_args()


def load_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def short_label(row):
    start = row["frame_start"]
    end = row["frame_end"]
    terrain = row["terrain_label"]
    short = {
        "Indoor corridor carpet": "Indoor corridor",
        "Threshold / indoor-to-outdoor hard floor": "Threshold",
        "Outdoor paved brick walkway": "Paved walkway",
        "Open paved plaza": "Paved plaza",
        "Pavement-grass edge": "Pavement-grass",
        "Grass-side path edge": "Grass-side path",
        "Grass / soil / leaf-litter foreground": "Grass/soil",
        "Paved slabs with leaf-litter edge": "Slabs + leaves",
        "Carpeted corridor": "Carpet corridor",
        "Tile floor": "Tile floor",
        "Brick pavement": "Brick pavement",
        "Brick pavement to grass transition": "Brick->grass",
        "Grass": "Grass",
        "Slabs pavement": "Slabs",
        "Gravel": "Gravel",
        "Brick pavement and grass mixture": "Brick+grass mix",
    }.get(terrain, terrain)
    return f"{start}-{end}\n{short}"


def float_array(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=float)


def plot_panels(rows, output_path: Path):
    labels = [short_label(row) for row in rows]
    x = np.arange(len(rows))

    vel_err_mean = float_array(rows, "vel_err_mean")
    vel_err_std = float_array(rows, "vel_err_std")
    vel_tau_mean = float_array(rows, "vel_tau_mean")
    vel_tau_std = float_array(rows, "vel_tau_std")

    force_err_mean = float_array(rows, "force_err_mean")
    force_err_std = float_array(rows, "force_err_std")
    force_tau_mean = float_array(rows, "force_tau_mean")
    force_tau_std = float_array(rows, "force_tau_std")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Terrain-Window Signal Comparison", fontsize=24, fontweight="bold")

    def finish_axis(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=11)

    ax = axes[0, 0]
    ax.bar(x, vel_err_mean, yerr=vel_err_std, capsize=4, color="#4C78A8", alpha=0.9)
    ax.set_title("Velocity Error by Terrain Window", fontsize=16)
    ax.set_ylabel("Filtered velocity error", fontsize=13)
    finish_axis(ax)

    ax = axes[0, 1]
    ax.plot(x, vel_tau_mean, marker="o", linewidth=2.5, color="#4C78A8")
    ax.fill_between(
        x,
        vel_tau_mean - vel_tau_std,
        vel_tau_mean + vel_tau_std,
        color="#4C78A8",
        alpha=0.18,
    )
    pad = max((vel_tau_mean.max() - vel_tau_mean.min()) * 0.25, 2e-5)
    ax.set_ylim(vel_tau_mean.min() - pad, vel_tau_mean.max() + pad)
    ax.set_title("Velocity Traversability by Terrain Window", fontsize=16)
    ax.set_ylabel("Traversability score τ", fontsize=13)
    finish_axis(ax)

    ax = axes[1, 0]
    ax.bar(x, force_err_mean, yerr=force_err_std, capsize=4, color="#F58518", alpha=0.9)
    ax.set_title("Force Error by Terrain Window", fontsize=16)
    ax.set_ylabel("Filtered force error", fontsize=13)
    finish_axis(ax)

    ax = axes[1, 1]
    ax.plot(x, force_tau_mean, marker="o", linewidth=2.5, color="#F58518")
    ax.fill_between(
        x,
        force_tau_mean - force_tau_std,
        force_tau_mean + force_tau_std,
        color="#F58518",
        alpha=0.18,
    )
    pad = max((force_tau_mean.max() - force_tau_mean.min()) * 0.15, 0.001)
    ax.set_ylim(force_tau_mean.min() - pad, force_tau_mean.max() + pad)
    ax.set_title("Force Traversability by Terrain Window", fontsize=16)
    ax.set_ylabel("Traversability score τ", fontsize=13)
    finish_axis(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_table(rows, output_path: Path):
    headers = [
        "Frames",
        "Terrain",
        "Vel err",
        "Vel τ",
        "Force err",
        "Force τ",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                f"{row['frame_start']}-{row['frame_end']}",
                row["terrain_label"],
                f"{float(row['vel_err_mean']):.6f}",
                f"{float(row['vel_tau_mean']):.6f}",
                f"{float(row['force_err_mean']):.6f}",
                f"{float(row['force_tau_mean']):.6f}",
            ]
        )

    fig_h = 1.4 + 0.55 * len(table_rows)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")
    ax.set_title("Terrain-Window Summary Table", fontsize=22, fontweight="bold", pad=20)

    col_widths = [0.12, 0.36, 0.12, 0.12, 0.14, 0.14]
    table = ax.table(
        cellText=table_rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#2F4B7C")
        else:
            cell.set_facecolor("#F6F8FB" if row_idx % 2 else "#E9EEF6")
            if col_idx == 1:
                cell.set_text_props(ha="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    rows = load_rows(Path(args.summary_csv))
    plot_panels(rows, Path(args.plot_output))
    render_table(rows, Path(args.table_output))
    print(f"[ok] wrote {args.plot_output}")
    print(f"[ok] wrote {args.table_output}")


if __name__ == "__main__":
    main()
