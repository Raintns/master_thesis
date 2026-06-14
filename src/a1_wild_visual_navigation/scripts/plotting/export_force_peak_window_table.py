#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a focused table for force-peak and variability analysis across terrain windows."
    )
    parser.add_argument(
        "--manifest",
        default="/home/rain/github_upload/Result/force_default_normalize/wvn_frames/manifest.csv",
    )
    parser.add_argument(
        "--force-csv",
        default="/home/rain/github_upload/Result/force_default_normalize/wvn_force_debug.csv",
    )
    parser.add_argument(
        "--csv-output",
        default="/home/rain/github_upload/Result/rendered_figures/force_peak_window_table.csv",
    )
    parser.add_argument(
        "--tex-output",
        default="/home/rain/github_upload/Result/rendered_figures/force_peak_window_table.tex",
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
    return sorted(pairs)


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
    out = []
    for row in rows:
        value = row.get("force_error_filtered", "")
        if value == "":
            continue
        ts = float(row["timestamp"])
        if include_end:
            ok = t0 <= ts <= t1
        else:
            ok = t0 <= ts < t1
        if ok:
            out.append((ts, float(value)))
    return out


def local_peak_count(signal, threshold):
    count = 0
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            count += 1
    return count


def short_label(label: str) -> str:
    mapping = {
        "Carpeted corridor": "Carpet corridor",
        "Tile floor": "Tile floor",
        "Brick pavement": "Brick pavement",
        "Grass": "Grass",
        "Slabs pavement": "Slabs",
        "Gravel": "Gravel",
        "Brick pavement and grass mixture": "Brick+grass mix",
    }
    return mapping.get(label, label)


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(rows, threshold, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Force-error variability and strong-peak statistics across terrain windows. Strong peaks are defined as local maxima of the filtered force error above the global 90th-percentile threshold "
        + rf"$\theta_{{0.90}}={threshold:.4f}$" + r".}",
        r"\label{tab:force_peak_window_stats}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Window & Mean & Std & Max & Above $\theta_{0.90}$ & Peak count & Peak rate (Hz) \\",
        r"\midrule",
    ]
    for row in rows:
        window = f"{row['frame_start']}-{row['frame_end']} {short_label(row['terrain_label'])}"
        lines.append(
            f"{window} & "
            f"{float(row['force_error_mean']):.4f} & "
            f"{float(row['force_error_std']):.4f} & "
            f"{float(row['force_error_max']):.4f} & "
            f"{int(row['count_above_global_p90'])} & "
            f"{int(row['peak_count'])} & "
            f"{float(row['peak_rate_hz']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    manifest_rows = load_csv(Path(args.manifest))
    force_rows = load_csv(Path(args.force_csv))
    pairs = frame_times(manifest_rows)

    all_force_error = np.array(
        [float(row["force_error_filtered"]) for row in force_rows if row.get("force_error_filtered", "") != ""],
        dtype=float,
    )
    threshold = float(np.quantile(all_force_error, 0.9))

    rows_out = []
    for idx, (frame_start, frame_end, terrain_label) in enumerate(WINDOWS):
        t0 = interpolate_timestamp(pairs, frame_start)
        t1 = interpolate_timestamp(pairs, frame_end)
        rows = select_rows(force_rows, t0, t1, include_end=(idx == len(WINDOWS) - 1))
        timestamps = np.array([row[0] for row in rows], dtype=float)
        force_error = np.array([row[1] for row in rows], dtype=float)
        duration = float(max(timestamps[-1] - timestamps[0], 0.0)) if len(timestamps) else 0.0
        peak_count = local_peak_count(force_error, threshold) if len(force_error) >= 3 else 0
        peak_rate_hz = float(peak_count / max(duration, 1e-9)) if duration > 0.0 else 0.0
        rows_out.append(
            {
                "terrain_label": terrain_label,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "samples": int(len(force_error)),
                "duration_sec": duration,
                "force_error_mean": float(force_error.mean()),
                "force_error_std": float(force_error.std()),
                "force_error_max": float(force_error.max()),
                "count_above_global_p90": int(np.sum(force_error > threshold)),
                "peak_count": int(peak_count),
                "peak_rate_hz": peak_rate_hz,
                "global_p90_threshold": threshold,
            }
        )

    write_csv(rows_out, Path(args.csv_output))
    write_tex(rows_out, threshold, Path(args.tex_output))
    print(f"[ok] wrote {args.csv_output}")
    print(f"[ok] wrote {args.tex_output}")
    print(f"[info] global 90th-percentile threshold = {threshold:.6f}")


if __name__ == "__main__":
    main()
