#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WINDOWS = [
    (420, 770, "Carpeted corridor"),
    (770, 1000, "Tile floor"),
    (1000, 2140, "Brick pavement"),
    (2140, 2780, "Grass"),
    (2780, 2820, "Slabs pavement"),
    (2820, 2900, "Gravel"),
    (2900, 3080, "Brick pavement and grass mixture"),
    (3080, 3340, "Brick pavement"),
]

LEGS = ["FL", "FR", "RL", "RR"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze aggregate force patterns across terrain windows.")
    parser.add_argument(
        "--manifest",
        default="/home/rain/github_upload/Result/force_default_normalize/wvn_frames/manifest.csv",
    )
    parser.add_argument(
        "--force-csv",
        default="/home/rain/github_upload/Result/force_default_normalize/wvn_force_debug.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="/home/rain/github_upload/Result/rendered_figures/aggregate_force_pattern_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        default="/home/rain/github_upload/Result/rendered_figures/aggregate_force_pattern_panels.png",
    )
    return parser.parse_args()


def load_csv(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def frame_times(manifest_rows):
    pairs = []
    for row in manifest_rows:
        frame = int(row["frame_index"])
        ts = int(row["stamp_secs"]) + int(row["stamp_nsecs"]) * 1e-9
        pairs.append((frame, ts))
    pairs.sort()
    return pairs


def interpolate_timestamp(frame_time_pairs, frame_index):
    if frame_index <= frame_time_pairs[0][0]:
        return frame_time_pairs[0][1]
    if frame_index >= frame_time_pairs[-1][0]:
        return frame_time_pairs[-1][1]
    for (f0, t0), (f1, t1) in zip(frame_time_pairs[:-1], frame_time_pairs[1:]):
        if f0 <= frame_index <= f1:
            if f1 == f0:
                return t0
            alpha = (frame_index - f0) / float(f1 - f0)
            return t0 + alpha * (t1 - t0)
    raise ValueError(f"Could not interpolate timestamp for frame {frame_index}")


def select_rows(rows, t0, t1, include_end=False):
    selected = []
    for row in rows:
        ts = float(row["timestamp"])
        if include_end:
            if t0 <= ts <= t1:
                selected.append(row)
        else:
            if t0 <= ts < t1:
                selected.append(row)
    return selected


def safe_float(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def short_label(label):
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


def local_peak_rate(signal, timestamps, threshold):
    if len(signal) < 3:
        return 0.0, 0
    count = 0
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            count += 1
    duration = max(timestamps[-1] - timestamps[0], 1e-9)
    return count / duration, count


def summarize_window(rows, global_peak_threshold):
    timestamps = []
    total_load = []
    diag_a_share = []
    diag_b_share = []
    force_error = []

    for row in rows:
        ts = float(row["timestamp"])
        leg_vals = []
        valid = True
        for leg in LEGS:
            val = safe_float(row, f"current_force_{leg}z")
            if val is None:
                valid = False
                break
            leg_vals.append(max(val, 0.0))
        fe = safe_float(row, "force_error_filtered")
        if not valid or fe is None:
            continue

        fl, fr, rl, rr = leg_vals
        total = fl + fr + rl + rr
        if total <= 1e-9:
            continue

        timestamps.append(ts)
        total_load.append(total)
        diag_a_share.append((fl + rr) / total)
        diag_b_share.append((fr + rl) / total)
        force_error.append(fe)

    timestamps = np.array(timestamps, dtype=float)
    total_load = np.array(total_load, dtype=float)
    diag_a_share = np.array(diag_a_share, dtype=float)
    diag_b_share = np.array(diag_b_share, dtype=float)
    force_error = np.array(force_error, dtype=float)

    if len(total_load) == 0:
        return None

    diff_total = np.diff(total_load)
    fluctuation = float(np.mean(np.abs(diff_total))) if len(diff_total) else 0.0
    peak_rate_hz, peak_count = local_peak_rate(force_error, timestamps, global_peak_threshold)

    return {
        "samples": int(len(total_load)),
        "duration_sec": float(max(timestamps[-1] - timestamps[0], 0.0)),
        "total_load_mean": float(total_load.mean()),
        "total_load_std": float(total_load.std()),
        "total_load_var": float(total_load.var()),
        "total_load_cv": float(total_load.std() / max(total_load.mean(), 1e-9)),
        "total_load_fluctuation_mean_abs_diff": fluctuation,
        "force_error_mean": float(force_error.mean()),
        "force_error_std": float(force_error.std()),
        "force_error_var": float(force_error.var()),
        "peak_rate_hz": float(peak_rate_hz),
        "peak_count": int(peak_count),
        "diag_FL_RR_share": float(diag_a_share.mean()),
        "diag_FR_RL_share": float(diag_b_share.mean()),
        "diag_share_imbalance": float(np.mean(np.abs(diag_a_share - diag_b_share))),
    }


def write_summary(rows_out, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def plot_summary(rows_out, path: Path):
    labels = [f"{r['frame_start']}-{r['frame_end']}\n{short_label(r['terrain_label'])}" for r in rows_out]
    x = np.arange(len(rows_out))

    total_mean = np.array([float(r["total_load_mean"]) for r in rows_out], dtype=float)
    total_std = np.array([float(r["total_load_std"]) for r in rows_out], dtype=float)
    err_std = np.array([float(r["force_error_std"]) for r in rows_out], dtype=float)
    peak_rate = np.array([float(r["peak_rate_hz"]) for r in rows_out], dtype=float)
    diag_imbalance = np.array([float(r["diag_share_imbalance"]) for r in rows_out], dtype=float)
    diag_a = np.array([float(r["diag_FL_RR_share"]) for r in rows_out], dtype=float)
    diag_b = np.array([float(r["diag_FR_RL_share"]) for r in rows_out], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Aggregate Force Patterns Across Terrain Windows", fontsize=24, fontweight="bold")

    ax = axes[0, 0]
    ax.bar(x, total_mean, yerr=total_std, capsize=4, color="#4C78A8", alpha=0.9)
    ax.set_title("Total Supported Vertical Load", fontsize=16)
    ax.set_ylabel("Mean total load (positive-clamped z)", fontsize=13)

    ax = axes[0, 1]
    ax.plot(x, err_std, marker="o", linewidth=2.5, color="#E67E22", label="Std of filtered force error")
    ax.plot(
        x,
        peak_rate,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#2CA02C",
        label="Peak rate of filtered force error",
    )
    ax.set_title("Force Variability and Peak Density", fontsize=16)
    ax.set_ylabel("Window statistic", fontsize=13)
    ax.legend(loc="best", fontsize=10)

    ax = axes[1, 0]
    ax.plot(x, diag_a, marker="o", linewidth=2.5, color="#7F3C8D", label="FL + RR share")
    ax.plot(x, diag_b, marker="o", linewidth=2.5, color="#11A579", label="FR + RL share")
    ax.set_title("Diagonal Pair Load Share", fontsize=16)
    ax.set_ylabel("Mean diagonal load share", fontsize=13)
    ax.set_ylim(0.42, 0.58)
    ax.legend(loc="best", fontsize=10)

    ax = axes[1, 1]
    ax.bar(x, diag_imbalance, color="#D62728", alpha=0.88)
    ax.set_title("Diagonal Share Imbalance", fontsize=16)
    ax.set_ylabel("Mean |(FL+RR) - (FR+RL)| share", fontsize=13)

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.tick_params(axis="y", labelsize=11)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    manifest_rows = load_csv(Path(args.manifest))
    force_rows = load_csv(Path(args.force_csv))
    f_times = frame_times(manifest_rows)

    global_force_error = np.array(
        [float(row["force_error_filtered"]) for row in force_rows if row.get("force_error_filtered", "") != ""],
        dtype=float,
    )
    global_peak_threshold = float(np.quantile(global_force_error, 0.90))

    rows_out = []
    for i, (frame_start, frame_end, label) in enumerate(WINDOWS):
        t0 = interpolate_timestamp(f_times, frame_start)
        t1 = interpolate_timestamp(f_times, frame_end)
        window_rows = select_rows(force_rows, t0, t1, include_end=(i == len(WINDOWS) - 1))
        summary = summarize_window(window_rows, global_peak_threshold)
        if summary is None:
            continue
        rows_out.append(
            {
                "terrain_label": label,
                "frame_start": frame_start,
                "frame_end": frame_end,
                **summary,
            }
        )

    write_summary(rows_out, Path(args.summary_output))
    plot_summary(rows_out, Path(args.figure_output))

    print(f"[ok] wrote {args.summary_output}")
    print(f"[ok] wrote {args.figure_output}")
    print("terrain|total_load_std|force_error_std|peak_rate_hz|diag_imbalance")
    for row in rows_out:
        print(
            f"{row['terrain_label']}|{row['total_load_std']:.4f}|{row['force_error_std']:.4f}|"
            f"{row['peak_rate_hz']:.4f}|{row['diag_share_imbalance']:.4f}"
        )


if __name__ == "__main__":
    main()
