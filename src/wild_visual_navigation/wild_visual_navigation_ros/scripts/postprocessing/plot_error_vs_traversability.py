#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.size"] = 9

FORCE_COLOR = "#c44e52"
VELOCITY_COLOR = "#4c72b0"


def safe_read_csv(path):
    return pd.read_csv(path)


def prepare_frame(df, error_col, instant_col, raw_error_col=None, frame_col=None, timestamp_col=None):
    keep_cols = [error_col, instant_col]
    if raw_error_col:
        keep_cols.append(raw_error_col)
    if frame_col:
        keep_cols.append(frame_col)
    if timestamp_col:
        keep_cols.append(timestamp_col)
    frame = df[keep_cols].apply(pd.to_numeric, errors="coerce").dropna()
    rename_map = {
        error_col: "filtered_error",
        instant_col: "instant_traversability",
    }
    if raw_error_col:
        rename_map[raw_error_col] = "raw_error"
    if frame_col:
        rename_map[frame_col] = "frame_index"
    if timestamp_col:
        rename_map[timestamp_col] = "timestamp"
    frame = frame.rename(columns=rename_map)
    if "timestamp" in frame.columns:
        frame["time_since_start"] = frame["timestamp"] - frame["timestamp"].iloc[0]
    return frame


def pearson_corr(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def quantile_binned_means(frame, bins=5):
    ranked = frame.copy()
    ranked["bin"] = pd.qcut(ranked["filtered_error"], q=bins, labels=False, duplicates="drop")
    grouped = ranked.groupby("bin", observed=True).agg(
        filtered_error_mean=("filtered_error", "mean"),
        filtered_error_min=("filtered_error", "min"),
        filtered_error_max=("filtered_error", "max"),
        instant_traversability_mean=("instant_traversability", "mean"),
        instant_traversability_std=("instant_traversability", "std"),
        count=("instant_traversability", "count"),
    )
    grouped["instant_traversability_std"] = grouped["instant_traversability_std"].fillna(0.0)
    grouped["bin_index"] = np.arange(1, len(grouped) + 1)
    return grouped.reset_index(drop=True)


def summarize(frame):
    q = frame.quantile([0.05, 0.5, 0.95])
    return {
        "count": int(len(frame)),
        "pearson_filtered_vs_instant": pearson_corr(
            frame["filtered_error"].to_numpy(), frame["instant_traversability"].to_numpy()
        ),
        "instant_traversability": {
            "min": float(frame["instant_traversability"].min()),
            "q05": float(q.loc[0.05, "instant_traversability"]),
            "median": float(q.loc[0.5, "instant_traversability"]),
            "q95": float(q.loc[0.95, "instant_traversability"]),
            "max": float(frame["instant_traversability"].max()),
            "std": float(frame["instant_traversability"].std(ddof=0)),
        },
        "filtered_error": {
            "min": float(frame["filtered_error"].min()),
            "q05": float(q.loc[0.05, "filtered_error"]),
            "median": float(q.loc[0.5, "filtered_error"]),
            "q95": float(q.loc[0.95, "filtered_error"]),
            "max": float(frame["filtered_error"].max()),
            "std": float(frame["filtered_error"].std(ddof=0)),
        },
    }


def add_scatter_panel(ax, frame, color, title):
    sample = frame if len(frame) <= 2500 else frame.sample(2500, random_state=0)
    ax.scatter(
        sample["filtered_error"],
        sample["instant_traversability"],
        s=8,
        alpha=0.18,
        color=color,
        edgecolors="none",
    )
    binned = quantile_binned_means(frame, bins=6)
    ax.plot(
        binned["filtered_error_mean"],
        binned["instant_traversability_mean"],
        color="black",
        linewidth=1.5,
        marker="o",
        markersize=4,
    )
    ax.set_title(title)
    ax.set_xlabel("Filtered error")
    ax.set_ylabel("Instant traversability")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)


def add_quintile_panel(ax, force_bins, velocity_bins):
    ax.plot(
        force_bins["bin_index"],
        force_bins["instant_traversability_mean"],
        color=FORCE_COLOR,
        marker="o",
        linewidth=2,
        label="Force",
    )
    ax.plot(
        velocity_bins["bin_index"],
        velocity_bins["instant_traversability_mean"],
        color=VELOCITY_COLOR,
        marker="o",
        linewidth=2,
        label="Velocity",
    )
    ax.set_xticks(force_bins["bin_index"])
    ax.set_xlabel("Error quintile (low to high)")
    ax.set_ylabel("Mean instant traversability")
    ax.set_title("Error Quintile Separation")
    ax.grid(True, alpha=0.25)
    ax.legend()


def add_distribution_panel(ax, force_frame, velocity_frame):
    force_sorted = np.sort(force_frame["instant_traversability"].to_numpy())
    velocity_sorted = np.sort(velocity_frame["instant_traversability"].to_numpy())
    force_cdf = np.linspace(0.0, 1.0, len(force_sorted))
    velocity_cdf = np.linspace(0.0, 1.0, len(velocity_sorted))
    ax.plot(force_sorted, force_cdf, color=FORCE_COLOR, linewidth=2, label="Force")
    ax.plot(velocity_sorted, velocity_cdf, color=VELOCITY_COLOR, linewidth=2, label="Velocity")
    ax.set_xlabel("Instant traversability")
    ax.set_ylabel("CDF")
    ax.set_title("Traversability Distribution")
    ax.grid(True, alpha=0.25)
    ax.legend()


def add_time_panel(ax, frame, color, title):
    ax.plot(
        frame["frame_index"],
        frame["instant_traversability"],
        color=color,
        linewidth=2,
        label="Instant traversability",
    )
    ax.set_title(title)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Instant traversability", color=color)
    ax.tick_params(axis="y", labelcolor=color)
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        frame["frame_index"],
        frame["filtered_error"],
        color="black",
        linewidth=1.3,
        alpha=0.75,
        label="Filtered error",
    )
    ax2.set_ylabel("Filtered error", color="black")
    ax2.tick_params(axis="y", labelcolor="black")


def add_normalized_time_panel(ax, force_frame, velocity_frame):
    force_norm = (force_frame["instant_traversability"] - force_frame["instant_traversability"].mean()) / (
        force_frame["instant_traversability"].std(ddof=0) + 1e-12
    )
    velocity_norm = (
        velocity_frame["instant_traversability"] - velocity_frame["instant_traversability"].mean()
    ) / (velocity_frame["instant_traversability"].std(ddof=0) + 1e-12)
    ax.plot(force_frame["frame_index"], force_norm, color=FORCE_COLOR, linewidth=2, label="Force")
    ax.plot(velocity_frame["frame_index"], velocity_norm, color=VELOCITY_COLOR, linewidth=2, label="Velocity")
    ax.set_title("Normalized Traversability Over Time")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Z-scored instant traversability")
    ax.grid(True, alpha=0.25)
    ax.legend()


def add_summary_text(ax, force_summary, velocity_summary, force_bins, velocity_bins):
    force_drop = float(force_bins["instant_traversability_mean"].iloc[0] - force_bins["instant_traversability_mean"].iloc[-1])
    velocity_drop = float(
        velocity_bins["instant_traversability_mean"].iloc[0] - velocity_bins["instant_traversability_mean"].iloc[-1]
    )
    ratio = float(force_drop / velocity_drop) if abs(velocity_drop) > 1e-12 else float("inf")
    text = "\n".join(
        [
            "Key takeaways",
            f"Force corr(error, instant) = {force_summary['pearson_filtered_vs_instant']:.3f}",
            f"Velocity corr(error, instant) = {velocity_summary['pearson_filtered_vs_instant']:.3f}",
            f"Force traversability std = {force_summary['instant_traversability']['std']:.5f}",
            f"Velocity traversability std = {velocity_summary['instant_traversability']['std']:.5f}",
            f"Force quintile drop = {force_drop:.5f}",
            f"Velocity quintile drop = {velocity_drop:.5f}",
            f"Force / velocity drop ratio = {ratio:.1f}x",
        ]
    )
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "#f5f5f5", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.5"},
    )
    return {
        "force_quintile_drop": force_drop,
        "velocity_quintile_drop": velocity_drop,
        "drop_ratio_force_over_velocity": ratio,
    }


def save_summary(output_path, force_summary, velocity_summary, comparison_summary, force_bins, velocity_bins):
    payload = {
        "force": force_summary,
        "velocity": velocity_summary,
        "comparison": comparison_summary,
        "force_quintiles": force_bins.to_dict(orient="records"),
        "velocity_quintiles": velocity_bins.to_dict(orient="records"),
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Plot WVN filtered error against instant traversability for force and velocity runs."
    )
    parser.add_argument("--force-csv", required=True, help="Path to wvn_force_debug.csv")
    parser.add_argument("--velocity-csv", required=True, help="Path to wvn_velocity_debug.csv")
    parser.add_argument("--force-manifest", help="Optional path to force frame manifest.csv")
    parser.add_argument("--velocity-manifest", help="Optional path to velocity frame manifest.csv")
    parser.add_argument(
        "--output-dir",
        default="Result/error_vs_traversability_plots",
        help="Directory for output plots and summary json.",
    )
    parser.add_argument("--start-frame", type=int, default=None, help="Optional first frame index to keep.")
    parser.add_argument("--end-frame", type=int, default=None, help="Optional last frame index to keep.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.force_manifest:
        force_df = safe_read_csv(args.force_manifest)
        force_frame = prepare_frame(
            force_df,
            error_col="force_error_filtered",
            raw_error_col="force_error_mse",
            instant_col="instant_traversability",
            frame_col="frame_index",
        )
    else:
        force_df = safe_read_csv(args.force_csv)
        force_frame = prepare_frame(
            force_df,
            error_col="force_error_filtered",
            raw_error_col="force_error_mse",
            instant_col="instant_traversability",
            timestamp_col="timestamp",
        )

    if args.velocity_manifest:
        velocity_df = safe_read_csv(args.velocity_manifest)
        velocity_frame = prepare_frame(
            velocity_df,
            error_col="velocity_error_filtered",
            raw_error_col="velocity_error_mse",
            instant_col="instant_traversability",
            frame_col="frame_index",
        )
    else:
        velocity_df = safe_read_csv(args.velocity_csv)
        velocity_frame = prepare_frame(
            velocity_df,
            error_col="velocity_tracking_error_filtered",
            raw_error_col="velocity_tracking_error_mse",
            instant_col="instant_traversability",
            timestamp_col="timestamp",
        )

    if args.start_frame is not None and "frame_index" in force_frame.columns:
        force_frame = force_frame[force_frame["frame_index"] >= args.start_frame]
    if args.end_frame is not None and "frame_index" in force_frame.columns:
        force_frame = force_frame[force_frame["frame_index"] <= args.end_frame]
    if args.start_frame is not None and "frame_index" in velocity_frame.columns:
        velocity_frame = velocity_frame[velocity_frame["frame_index"] >= args.start_frame]
    if args.end_frame is not None and "frame_index" in velocity_frame.columns:
        velocity_frame = velocity_frame[velocity_frame["frame_index"] <= args.end_frame]

    force_frame = force_frame.reset_index(drop=True)
    velocity_frame = velocity_frame.reset_index(drop=True)

    force_summary = summarize(force_frame)
    velocity_summary = summarize(velocity_frame)
    force_bins = quantile_binned_means(force_frame, bins=5)
    velocity_bins = quantile_binned_means(velocity_frame, bins=5)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=180, constrained_layout=True)
    add_scatter_panel(axes[0, 0], force_frame, FORCE_COLOR, "Force: Filtered Error vs Instant Traversability")
    add_scatter_panel(
        axes[0, 1], velocity_frame, VELOCITY_COLOR, "Velocity: Filtered Error vs Instant Traversability"
    )
    add_quintile_panel(axes[0, 2], force_bins, velocity_bins)
    add_time_panel(axes[1, 0], force_frame, FORCE_COLOR, "Force Over Time")
    add_time_panel(axes[1, 1], velocity_frame, VELOCITY_COLOR, "Velocity Over Time")
    comparison_summary = add_summary_text(
        axes[1, 2], force_summary, velocity_summary, force_bins, velocity_bins
    )

    if args.start_frame is not None or args.end_frame is not None:
        comparison_summary["frame_window"] = {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
        }

    output_png = output_dir / "force_vs_velocity_error_traversability.png"
    output_pdf = output_dir / "force_vs_velocity_error_traversability.pdf"
    output_json = output_dir / "force_vs_velocity_error_traversability_summary.json"
    fig.savefig(output_png)
    fig.savefig(output_pdf)
    plt.close(fig)

    force_bins.to_csv(output_dir / "force_quintiles.csv", index=False)
    velocity_bins.to_csv(output_dir / "velocity_quintiles.csv", index=False)
    save_summary(output_json, force_summary, velocity_summary, comparison_summary, force_bins, velocity_bins)

    print(f"[ok] wrote {output_png}")
    print(f"[ok] wrote {output_pdf}")
    print(f"[ok] wrote {output_json}")


if __name__ == "__main__":
    main()
