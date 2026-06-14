#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LEG_COLORS = {
    "FL": "#c44e52",
    "FR": "#4c72b0",
    "RL": "#55a868",
    "RR": "#8172b3",
}


def load_force_data(csv_path, prefix, axis):
    df = pd.read_csv(csv_path)
    leg_cols = [f"{prefix}_{leg}{axis}" for leg in ("FL", "FR", "RL", "RR")]
    keep_cols = ["timestamp"] + leg_cols
    frame = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["timestamp"], how="any")
    frame["second"] = frame["timestamp"].astype(int)
    grouped = frame.groupby("second", observed=True)[leg_cols].mean().reset_index()
    grouped["time_since_start_s"] = grouped["second"] - grouped["second"].iloc[0]
    return grouped, leg_cols


def plot_leg(grouped, leg, col, output_path, title):
    fig, ax = plt.subplots(figsize=(10, 4), dpi=180, constrained_layout=True)
    leg_name = leg[:2]
    ax.plot(
        grouped["time_since_start_s"],
        grouped[col],
        color=LEG_COLORS[leg_name],
        linewidth=2,
    )
    ax.set_title(title)
    ax.set_xlabel("Time since start [s]")
    ax.set_ylabel("Force [N]")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot one per-leg force figure using one averaged sample per second."
    )
    parser.add_argument("--csv", required=True, help="Input debug CSV, e.g. wvn_force_debug.csv")
    parser.add_argument(
        "--prefix",
        default="raw_current_force",
        help="Column prefix, e.g. raw_current_force, raw_desired_force, current_force, desired_force",
    )
    parser.add_argument("--axis", default="z", help="Axis suffix, usually x, y, or z")
    parser.add_argument(
        "--output-dir",
        default="Result/leg_force_plots",
        help="Directory for output plots and the resampled CSV.",
    )
    args = parser.parse_args()

    plt.rcParams["font.size"] = 10

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axis = args.axis.lower()
    grouped, leg_cols = load_force_data(args.csv, args.prefix, axis)
    grouped.to_csv(output_dir / f"{args.prefix}_{args.axis.lower()}_per_second.csv", index=False)

    for col in leg_cols:
        leg = col.split("_")[-1]
        title = f"{args.prefix} {leg} averaged to 1 sample/s"
        base = f"{args.prefix}_{leg}_per_second"
        plot_leg(grouped, leg, col, output_dir / f"{base}.png", title)
        plot_leg(grouped, leg, col, output_dir / f"{base}.pdf", title)

    print(f"[ok] wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
