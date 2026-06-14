#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis-friendly terrain-window comparison assets."
    )
    parser.add_argument(
        "--summary-csv",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_signal_summary.csv",
        help="Terrain-window summary CSV.",
    )
    parser.add_argument(
        "--compact-output",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_compact_comparison.png",
        help="Output path for compact 2-panel comparison figure.",
    )
    parser.add_argument(
        "--heatmap-output",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_signal_heatmap.png",
        help="Output path for normalized heatmap figure.",
    )
    parser.add_argument(
        "--latex-output",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_signal_table.tex",
        help="Output path for LaTeX table.",
    )
    return parser.parse_args()


def load_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def short_label(row):
    terrain = row["terrain_label"]
    mapping = {
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
    }
    return mapping.get(terrain, terrain)


def to_float_array(rows, key):
    return np.array([float(row[key]) for row in rows], dtype=float)


def normalize(values):
    min_v = values.min()
    max_v = values.max()
    if np.isclose(max_v, min_v):
        return np.zeros_like(values)
    return (values - min_v) / (max_v - min_v)


def plot_compact(rows, output_path: Path):
    labels = [short_label(row) for row in rows]
    frame_ranges = [f"{row['frame_start']}-{row['frame_end']}" for row in rows]
    x = np.arange(len(rows))

    vel_tau = to_float_array(rows, "vel_tau_mean")
    force_tau = to_float_array(rows, "force_tau_mean")
    vel_tau_std = to_float_array(rows, "vel_tau_std")
    force_tau_std = to_float_array(rows, "force_tau_std")

    vel_err_norm = normalize(to_float_array(rows, "vel_err_mean"))
    force_err_norm = normalize(to_float_array(rows, "force_err_mean"))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Terrain-Window Comparison", fontsize=24, fontweight="bold")

    axes[0].plot(x, vel_tau, marker="o", linewidth=2.5, color="#3366AA", label="Velocity-based tau")
    axes[0].plot(x, force_tau, marker="o", linewidth=2.5, color="#E67E22", label="Force-based tau")
    axes[0].fill_between(x, vel_tau - vel_tau_std, vel_tau + vel_tau_std, color="#3366AA", alpha=0.15)
    axes[0].fill_between(
        x, force_tau - force_tau_std, force_tau + force_tau_std, color="#E67E22", alpha=0.15
    )
    axes[0].set_ylabel("Traversability score $\\tau$", fontsize=13)
    axes[0].set_title("Mean Traversability by Terrain Window", fontsize=16)
    axes[0].legend(loc="best", fontsize=11)

    axes[1].plot(
        x,
        vel_err_norm,
        marker="o",
        linewidth=2.5,
        color="#3366AA",
        label="Velocity error (min-max normalized)",
    )
    axes[1].plot(
        x,
        force_err_norm,
        marker="o",
        linewidth=2.5,
        color="#E67E22",
        label="Force error (min-max normalized)",
    )
    axes[1].set_ylabel("Normalized filtered error", fontsize=13)
    axes[1].set_title("Relative Error Variation Across Terrain Windows", fontsize=16)
    axes[1].legend(loc="best", fontsize=11)
    axes[1].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{frames}\n{label}" for frames, label in zip(frame_ranges, labels)], rotation=20, ha="right")
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows, output_path: Path):
    labels = [f"{row['frame_start']}-{row['frame_end']}\n{short_label(row)}" for row in rows]
    metrics = {
        "Velocity error": to_float_array(rows, "vel_err_mean"),
        "Velocity tau": to_float_array(rows, "vel_tau_mean"),
        "Force error": to_float_array(rows, "force_err_mean"),
        "Force tau": to_float_array(rows, "force_tau_mean"),
    }
    normalized = np.vstack([normalize(values) for values in metrics.values()])
    raw = np.vstack(list(metrics.values()))

    fig, ax = plt.subplots(figsize=(16, 5.5), constrained_layout=True)
    im = ax.imshow(normalized, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title("Normalized Terrain-Window Signal Heatmap", fontsize=22, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(list(metrics.keys()), fontsize=12)

    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            value = raw[i, j]
            text = f"{value:.6f}" if "tau" in list(metrics.keys())[i].lower() else f"{value:.6f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label("Row-wise normalized magnitude", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def escape_latex(text: str):
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_table(rows, output_path: Path):
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Terrain-window comparison of velocity-based and force-based supervision signals.}",
        "\\label{tab:terrain_window_signal_summary}",
        "\\begin{tabular}{p{1.8cm} p{5.8cm} c c c c}",
        "\\hline",
        "Frames & Terrain & Vel. err & Vel. $\\tau$ & Force err & Force $\\tau$ \\\\",
        "\\hline",
    ]

    for row in rows:
        lines.append(
            "{} & {} & {:.6f} & {:.6f} & {:.6f} & {:.6f} \\\\".format(
                escape_latex(f"{row['frame_start']}-{row['frame_end']}"),
                escape_latex(row["terrain_label"]),
                float(row["vel_err_mean"]),
                float(row["vel_tau_mean"]),
                float(row["force_err_mean"]),
                float(row["force_tau_mean"]),
            )
        )

    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    rows = load_rows(Path(args.summary_csv))
    plot_compact(rows, Path(args.compact_output))
    plot_heatmap(rows, Path(args.heatmap_output))
    write_latex_table(rows, Path(args.latex_output))
    print(f"[ok] wrote {args.compact_output}")
    print(f"[ok] wrote {args.heatmap_output}")
    print(f"[ok] wrote {args.latex_output}")


if __name__ == "__main__":
    main()
