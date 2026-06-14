#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("/home/rain/github_upload/scalar_timeseries.csv")
DEFAULT_OUTPUT_DIR = Path("/home/rain/github_upload/Result/rendered_figures")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a robust summary table for WVN scalar_timeseries.csv."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def fmt_sci(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${float(mantissa):.2f} \\\\times 10^{{{int(exponent)}}}$"


def fmt_float(value, digits=6):
    return f"{value:.{digits}f}"


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input)
    cols = [
        "sample_index",
        "elapsed_sec",
        "instant_traversability_local",
        "velocity_error_mse_local",
        "velocity_error_filtered_local",
    ]
    frame = frame[cols].dropna().copy()

    tau = frame["instant_traversability_local"].to_numpy(dtype=float)
    err_mse = frame["velocity_error_mse_local"].to_numpy(dtype=float)
    err_filtered = frame["velocity_error_filtered_local"].to_numpy(dtype=float)

    summary = {
        "samples": len(frame),
        "elapsed_start": float(frame["elapsed_sec"].min()),
        "elapsed_end": float(frame["elapsed_sec"].max()),
        "mse_mean": float(err_mse.mean()),
        "mse_std": float(err_mse.std(ddof=0)),
        "filtered_mean": float(err_filtered.mean()),
        "filtered_std": float(err_filtered.std(ddof=0)),
        "filtered_p05": float(np.quantile(err_filtered, 0.05)),
        "filtered_median": float(np.quantile(err_filtered, 0.50)),
        "filtered_p95": float(np.quantile(err_filtered, 0.95)),
        "tau_mean": float(tau.mean()),
        "tau_std": float(tau.std(ddof=0)),
        "tau_p05": float(np.quantile(tau, 0.05)),
        "tau_median": float(np.quantile(tau, 0.50)),
        "tau_p95": float(np.quantile(tau, 0.95)),
        "corr_filtered_tau": float(np.corrcoef(err_filtered, tau)[0, 1]),
    }

    csv_path = args.output_dir / "wvn_scalar_summary.csv"
    tex_path = args.output_dir / "wvn_scalar_summary.tex"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    tex = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Robust quantitative summary of the original WVN scalar time-series run.}}
\\label{{tab:wvn_scalar_summary}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Number of samples $n$ & {summary['samples']} \\\\
Elapsed time range & {summary['elapsed_start']:.2f} s to {summary['elapsed_end']:.2f} s \\\\
Mean velocity error MSE & {fmt_sci(summary['mse_mean'])} \\\\
Std.\\ velocity error MSE & {fmt_sci(summary['mse_std'])} \\\\
Mean filtered velocity error & {fmt_sci(summary['filtered_mean'])} \\\\
Std.\\ filtered velocity error & {fmt_sci(summary['filtered_std'])} \\\\
5th percentile filtered velocity error & {fmt_sci(summary['filtered_p05'])} \\\\
Median filtered velocity error & {fmt_sci(summary['filtered_median'])} \\\\
95th percentile filtered velocity error & {fmt_sci(summary['filtered_p95'])} \\\\
Mean traversability & {fmt_float(summary['tau_mean'])} \\\\
Std.\\ traversability & {fmt_sci(summary['tau_std'])} \\\\
5th percentile traversability & {fmt_float(summary['tau_p05'])} \\\\
Median traversability & {fmt_float(summary['tau_median'])} \\\\
95th percentile traversability & {fmt_float(summary['tau_p95'])} \\\\
Correlation (filtered error, traversability) & ${summary['corr_filtered_tau']:.5f}$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    tex_path.write_text(tex)

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {tex_path}")


if __name__ == "__main__":
    main()
