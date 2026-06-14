#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("/home/rain/github_upload/scalar_timeseries.csv")
DEFAULT_OUTPUT_DIR = Path("/home/rain/github_upload/Result/rendered_figures")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot WVN scalar_timeseries.csv in the same visual style as the A1 traversability/error figures."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tau-window", type=int, default=101)
    parser.add_argument("--error-window", type=int, default=33)
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to the output filenames, e.g. '_50samples'.",
    )
    return parser.parse_args()


def rolling_mean(values, window):
    if window <= 1:
        return values
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def load_scalar_series(path):
    frame = pd.read_csv(path)
    frame = frame.sort_values("sample_index").reset_index(drop=True)
    return frame


def build_basic_plot(frame):
    x = frame["sample_index"].to_numpy()
    tau = frame["instant_traversability_local"].to_numpy()
    err_raw = frame["velocity_error_mse_local"].to_numpy()
    err_filtered = frame["velocity_error_filtered_local"].to_numpy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.08]},
    )
    fig.suptitle("WVN Traversability and Velocity Error Over Time", fontsize=18, fontweight="normal")

    top_ax, bottom_ax = axes

    top_ax.plot(
        x,
        tau,
        color="#1f77b4",
        linewidth=1.6,
        label="Instant Traversability",
    )
    top_ax.set_title("Instant Traversability", fontsize=14)
    top_ax.set_ylabel("Traversability score τ")
    top_ax.grid(True, which="major", alpha=0.45)
    top_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    top_ax.minorticks_on()
    top_ax.legend(loc="lower left", frameon=True)

    bottom_ax.plot(
        x,
        err_raw,
        color="#ff7f0e",
        linewidth=1.4,
        label="Velocity error MSE",
    )
    bottom_ax.plot(
        x,
        err_filtered,
        color="#2ca02c",
        linewidth=1.6,
        linestyle="--",
        label="Velocity error filtered",
    )
    bottom_ax.set_title("Velocity error", fontsize=14)
    bottom_ax.set_xlabel("Sample Index")
    bottom_ax.set_ylabel("Velocity error")
    bottom_ax.grid(True, which="major", alpha=0.45)
    bottom_ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    bottom_ax.minorticks_on()
    bottom_ax.legend(loc="upper left", frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def build_combined_plot(frame, tau_window, error_window):
    x = frame["sample_index"].to_numpy()
    tau = frame["instant_traversability_local"].to_numpy()
    err_filtered = frame["velocity_error_filtered_local"].to_numpy()

    tau_trend = rolling_mean(tau, tau_window)
    err_trend = rolling_mean(err_filtered, error_window)

    trend_min = float(tau_trend.min())
    trend_max = float(tau_trend.max())
    trend_pad = max((trend_max - trend_min) * 0.12, 1e-5)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.5, 9.2),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1], "hspace": 0.16},
    )
    fig.suptitle("WVN Traversability and Velocity Error Over Time", fontsize=18, fontweight="normal")

    top_ax, mid_ax, bottom_ax = axes
    for ax in axes:
        ax.grid(True, which="major", alpha=0.28)
        ax.grid(True, which="minor", alpha=0.12, linestyle=":")
        ax.minorticks_on()

    top_ax.plot(
        x,
        tau,
        color="#96a0ad",
        linewidth=1.05,
        alpha=0.8,
        label="Instant traversability (raw)",
    )
    top_ax.plot(
        x,
        tau_trend,
        color="#1f77b4",
        linewidth=2.0,
        label="Instant traversability (trend)",
    )
    top_ax.set_title("Instant traversability", fontsize=13)
    top_ax.set_ylabel("Traversability score τ")
    top_ax.legend(loc="lower left", frameon=True)

    mid_ax.plot(
        x,
        tau,
        color="#96a0ad",
        linewidth=0.95,
        alpha=0.72,
        label="Instant traversability (raw)",
    )
    mid_ax.plot(
        x,
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
        x,
        err_filtered,
        color="#f28e2b",
        linewidth=1.05,
        alpha=0.62,
        label="Velocity error filtered (raw)",
    )
    bottom_ax.plot(
        x,
        err_trend,
        color="#2ca02c",
        linewidth=2.0,
        linestyle="--",
        label="Velocity error filtered (trend)",
    )
    bottom_ax.set_title("Velocity error filtered", fontsize=13)
    bottom_ax.set_xlabel("Sample Index")
    bottom_ax.set_ylabel("Velocity error")
    bottom_ax.legend(loc="upper left", frameon=True)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return fig


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_scalar_series(args.input)
    suffix = args.suffix

    basic = build_basic_plot(frame)
    combined = build_combined_plot(frame, args.tau_window, args.error_window)

    basic_path = args.output_dir / f"wvn_scalar_velocity_curves_like_a1{suffix}.png"
    combined_path = args.output_dir / f"wvn_scalar_velocity_curves_like_a1_combined{suffix}.png"

    basic.savefig(basic_path, facecolor="white")
    combined.savefig(combined_path, facecolor="white")
    plt.close(basic)
    plt.close(combined)

    print(f"[ok] wrote {basic_path}")
    print(f"[ok] wrote {combined_path}")


if __name__ == "__main__":
    main()
