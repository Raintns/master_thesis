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
    parser = argparse.ArgumentParser(description="Analyze per-leg force patterns across terrain windows.")
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
        default="/home/rain/github_upload/Result/rendered_figures/force_leg_pattern_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        default="/home/rain/github_upload/Result/rendered_figures/force_leg_pattern_heatmaps.png",
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


def values_for_leg(rows, prefix, leg):
    key = f"{prefix}_{leg}z"
    return np.array([float(row[key]) for row in rows if row.get(key, "") != ""], dtype=float)


def load_share_rows(rows):
    share_rows = []
    keys = [f"current_force_{leg}z" for leg in LEGS]
    for row in rows:
        if any(row.get(k, "") == "" for k in keys):
            continue
        vals = np.array([max(float(row[k]), 0.0) for k in keys], dtype=float)
        total = float(vals.sum())
        if total <= 1e-9:
            continue
        share_rows.append(vals / total)
    if not share_rows:
        return np.zeros((0, 4), dtype=float)
    return np.vstack(share_rows)


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


def write_summary(rows_out, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)


def plot_heatmaps(rows_out, path: Path):
    labels = [f"{r['frame_start']}-{r['frame_end']}\n{short_label(r['terrain_label'])}" for r in rows_out]

    mae = np.array([[float(r[f"{leg}_mae"]) for r in rows_out] for leg in LEGS], dtype=float)
    bias = np.array([[float(r[f"{leg}_bias"]) for r in rows_out] for leg in LEGS], dtype=float)
    shares = np.array([[float(r[f"{leg}_share"]) for r in rows_out] for leg in LEGS], dtype=float)

    front_share = np.array([float(r["front_share"]) for r in rows_out], dtype=float)
    rear_share = np.array([float(r["rear_share"]) for r in rows_out], dtype=float)
    left_share = np.array([float(r["left_share"]) for r in rows_out], dtype=float)
    right_share = np.array([float(r["right_share"]) for r in rows_out], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    fig.suptitle("Per-Leg Force Patterns Across Terrain Windows", fontsize=24, fontweight="bold")

    def draw_matrix(ax, matrix, title, cmap, fmt="{:.3f}", vmin=None, vmax=None):
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=16)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_yticks(np.arange(len(LEGS)))
        ax.set_yticklabels(LEGS, fontsize=12)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.86)

    draw_matrix(axes[0, 0], mae, "Per-Leg Mean Absolute Force Error", "YlOrRd")
    bias_lim = np.max(np.abs(bias))
    draw_matrix(
        axes[0, 1],
        bias,
        "Per-Leg Signed Force Error Bias",
        "coolwarm",
        vmin=-bias_lim,
        vmax=bias_lim,
    )
    draw_matrix(axes[1, 0], shares, "Per-Leg Mean Load Share", "Blues")

    ax = axes[1, 1]
    x = np.arange(len(labels))
    ax.plot(x, front_share, marker="o", linewidth=2.5, label="Front share", color="#E67E22")
    ax.plot(x, rear_share, marker="o", linewidth=2.5, label="Rear share", color="#1F77B4")
    ax.plot(x, left_share, marker="s", linewidth=2.0, linestyle="--", label="Left share", color="#2CA02C")
    ax.plot(x, right_share, marker="s", linewidth=2.0, linestyle="--", label="Right share", color="#D62728")
    ax.set_title("Front/Rear and Left/Right Load Distribution", fontsize=16)
    ax.set_ylabel("Load share", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0.35, 0.65)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.35)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    manifest_rows = load_csv(Path(args.manifest))
    force_rows = load_csv(Path(args.force_csv))
    f_times = frame_times(manifest_rows)

    rows_out = []
    for i, (frame_start, frame_end, label) in enumerate(WINDOWS):
        t0 = interpolate_timestamp(f_times, frame_start)
        t1 = interpolate_timestamp(f_times, frame_end)
        window_rows = select_rows(force_rows, t0, t1, include_end=(i == len(WINDOWS) - 1))

        shares = load_share_rows(window_rows)
        share_mean = shares.mean(axis=0) if len(shares) else np.zeros(4)

        row = {
            "terrain_label": label,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "samples": len(window_rows),
        }

        for leg_idx, leg in enumerate(LEGS):
            vals = values_for_leg(window_rows, "force_error", leg)
            abs_vals = np.abs(vals)
            row[f"{leg}_mae"] = float(abs_vals.mean()) if len(abs_vals) else 0.0
            row[f"{leg}_bias"] = float(vals.mean()) if len(vals) else 0.0
            row[f"{leg}_std"] = float(vals.std()) if len(vals) else 0.0
            row[f"{leg}_share"] = float(share_mean[leg_idx]) if len(shares) else 0.0

        row["front_share"] = row["FL_share"] + row["FR_share"]
        row["rear_share"] = row["RL_share"] + row["RR_share"]
        row["left_share"] = row["FL_share"] + row["RL_share"]
        row["right_share"] = row["FR_share"] + row["RR_share"]
        row["front_rear_bias"] = row["front_share"] - row["rear_share"]
        row["left_right_bias"] = row["left_share"] - row["right_share"]

        dominant_leg = max(LEGS, key=lambda leg: row[f"{leg}_mae"])
        row["dominant_error_leg"] = dominant_leg
        rows_out.append(row)

    write_summary(rows_out, Path(args.summary_output))
    plot_heatmaps(rows_out, Path(args.figure_output))

    print(f"[ok] wrote {args.summary_output}")
    print(f"[ok] wrote {args.figure_output}")
    print("terrain|dominant_error_leg|FL_mae|FR_mae|RL_mae|RR_mae|front_share|left_share")
    for row in rows_out:
        print(
            f"{row['terrain_label']}|{row['dominant_error_leg']}|"
            f"{row['FL_mae']:.4f}|{row['FR_mae']:.4f}|{row['RL_mae']:.4f}|{row['RR_mae']:.4f}|"
            f"{row['front_share']:.3f}|{row['left_share']:.3f}"
        )


if __name__ == "__main__":
    main()
