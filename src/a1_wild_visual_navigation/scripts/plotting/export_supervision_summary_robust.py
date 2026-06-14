#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export robust supervision summaries for velocity and force over a frame-index window."
        )
    )
    parser.add_argument(
        "--velocity-manifest",
        type=Path,
        default=Path(
            "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames/manifest.csv"
        ),
    )
    parser.add_argument(
        "--force-manifest",
        type=Path,
        default=Path("/home/rain/github_upload/Result/force_default_normalize/wvn_frames/manifest.csv"),
    )
    parser.add_argument(
        "--velocity-csv",
        type=Path,
        default=Path(
            "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_velocity_debug.csv"
        ),
    )
    parser.add_argument(
        "--force-csv",
        type=Path,
        default=Path("/home/rain/github_upload/Result/force_default_normalize/wvn_force_debug.csv"),
    )
    parser.add_argument("--range-start", type=int, default=420)
    parser.add_argument("--range-end", type=int, default=3340)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("/home/rain/github_upload/Result/rendered_figures/supervision_summary_420_3340_robust.csv"),
    )
    parser.add_argument(
        "--tex-output",
        type=Path,
        default=Path("/home/rain/github_upload/Result/rendered_figures/supervision_summary_420_3340_robust.tex"),
    )
    return parser.parse_args()


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


def paired_summary(debug_csv, manifest_csv, err_col, tau_col, frame_start, frame_end):
    manifest = load_manifest(manifest_csv)
    t0, t1 = frame_range_timestamps(manifest, frame_start, frame_end)
    frame = pd.read_csv(debug_csv)
    frame = frame[(frame["timestamp"] >= t0) & (frame["timestamp"] <= t1)][[err_col, tau_col]].dropna().copy()

    err = frame[err_col].to_numpy(dtype=float)
    tau = frame[tau_col].to_numpy(dtype=float)

    return {
        "samples": len(frame),
        "error_mean": float(err.mean()),
        "error_std": float(err.std(ddof=0)),
        "error_p05": float(np.quantile(err, 0.05)),
        "error_median": float(np.quantile(err, 0.50)),
        "error_p95": float(np.quantile(err, 0.95)),
        "tau_mean": float(tau.mean()),
        "tau_std": float(tau.std(ddof=0)),
        "tau_p05": float(np.quantile(tau, 0.05)),
        "tau_median": float(np.quantile(tau, 0.50)),
        "tau_p95": float(np.quantile(tau, 0.95)),
        "corr": float(np.corrcoef(err, tau)[0, 1]),
    }


def fmt_float(value, digits=6):
    return f"{value:.{digits}f}"


def fmt_sci(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${float(mantissa):.2f} \\times 10^{{{int(exponent)}}}$"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "modality",
        "samples",
        "error_mean",
        "error_std",
        "error_p05",
        "error_median",
        "error_p95",
        "tau_mean",
        "tau_std",
        "tau_p05",
        "tau_median",
        "tau_p95",
        "corr",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_table(modality, stats, frame_start, frame_end):
    is_velocity = modality == "velocity"
    error_name = "velocity" if is_velocity else "force"
    caption = (
        f"Robust quantitative summary of {modality}-based supervision results "
        f"for the frame window {frame_start}--{frame_end}."
    )
    label = f"tab:{modality}_summary_{frame_start}_{frame_end}_robust"
    error_mean_fmt = fmt_sci(stats["error_mean"]) if is_velocity else fmt_float(stats["error_mean"], 5)
    error_std_fmt = fmt_sci(stats["error_std"]) if is_velocity else fmt_float(stats["error_std"], 5)
    error_p05_fmt = fmt_sci(stats["error_p05"]) if is_velocity else fmt_float(stats["error_p05"], 5)
    error_median_fmt = fmt_sci(stats["error_median"]) if is_velocity else fmt_float(stats["error_median"], 5)
    error_p95_fmt = fmt_sci(stats["error_p95"]) if is_velocity else fmt_float(stats["error_p95"], 5)
    tau_std_fmt = fmt_sci(stats["tau_std"]) if is_velocity else fmt_float(stats["tau_std"], 5)
    corr_fmt = f"${stats['corr']:.5f}$"

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Value} \\\\",
        "\\midrule",
        f"Number of paired samples $n$ & {stats['samples']} \\\\",
        f"Mean filtered {error_name} error & {error_mean_fmt} \\\\",
        f"Std.\\ filtered {error_name} error & {error_std_fmt} \\\\",
        f"5th percentile filtered {error_name} error & {error_p05_fmt} \\\\",
        f"Median filtered {error_name} error & {error_median_fmt} \\\\",
        f"95th percentile filtered {error_name} error & {error_p95_fmt} \\\\",
        f"Mean traversability & {stats['tau_mean']:.6f} \\\\",
        f"Std.\\ traversability & {tau_std_fmt} \\\\",
        f"5th percentile traversability & {stats['tau_p05']:.6f} \\\\",
        f"Median traversability & {stats['tau_median']:.6f} \\\\",
        f"95th percentile traversability & {stats['tau_p95']:.6f} \\\\",
        f"Correlation (error, traversability) & {corr_fmt} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()

    velocity = paired_summary(
        args.velocity_csv,
        args.velocity_manifest,
        "velocity_tracking_error_filtered",
        "supervision_traversability",
        args.range_start,
        args.range_end,
    )
    force = paired_summary(
        args.force_csv,
        args.force_manifest,
        "force_error_filtered",
        "supervision_traversability",
        args.range_start,
        args.range_end,
    )

    rows = [
        {"modality": "velocity", **velocity},
        {"modality": "force", **force},
    ]
    write_csv(args.csv_output, rows)

    tex_content = "\n\n".join(
        [
            build_table("velocity", velocity, args.range_start, args.range_end),
            build_table("force", force, args.range_start, args.range_end),
        ]
    )
    args.tex_output.parent.mkdir(parents=True, exist_ok=True)
    args.tex_output.write_text(tex_content + "\n")

    print(f"[ok] wrote {args.csv_output}")
    print(f"[ok] wrote {args.tex_output}")


if __name__ == "__main__":
    main()
