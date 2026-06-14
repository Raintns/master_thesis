#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


DEFAULT_WINDOWS = [
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
        description="Compute terrain-window summaries from full debug CSV streams."
    )
    parser.add_argument(
        "--velocity-manifest",
        default="/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames/manifest.csv",
    )
    parser.add_argument(
        "--force-manifest",
        default="/home/rain/github_upload/Result/force_default_normalize/wvn_frames/manifest.csv",
    )
    parser.add_argument(
        "--velocity-csv",
        default="/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_velocity_debug.csv",
    )
    parser.add_argument(
        "--force-csv",
        default="/home/rain/github_upload/Result/force_default_normalize/wvn_force_debug.csv",
    )
    parser.add_argument(
        "--output",
        default="/home/rain/github_upload/Result/rendered_figures/terrain_window_signal_summary.csv",
    )
    return parser.parse_args()


def load_csv(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def manifest_frame_times(rows):
    pairs = []
    for row in rows:
        frame_index = int(row["frame_index"])
        stamp = int(row["stamp_secs"]) + int(row["stamp_nsecs"]) * 1e-9
        pairs.append((frame_index, stamp))
    pairs.sort()
    return pairs


def interpolate_timestamp(frame_times, frame_index):
    if frame_index <= frame_times[0][0]:
        return frame_times[0][1]
    if frame_index >= frame_times[-1][0]:
        return frame_times[-1][1]

    for (f0, t0), (f1, t1) in zip(frame_times[:-1], frame_times[1:]):
        if f0 <= frame_index <= f1:
            if f1 == f0:
                return t0
            alpha = (frame_index - f0) / float(f1 - f0)
            return t0 + alpha * (t1 - t0)

    raise ValueError(f"Could not interpolate frame index {frame_index}")


def stats(values):
    vals = [float(v) for v in values if v not in ("", None)]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "var": var,
        "min": min(vals),
        "max": max(vals),
    }


def select_rows(rows, t0, t1, include_end=False):
    out = []
    for row in rows:
        ts = float(row["timestamp"])
        if include_end:
            if t0 <= ts <= t1:
                out.append(row)
        else:
            if t0 <= ts < t1:
                out.append(row)
    return out


def main():
    args = parse_args()

    velocity_manifest = load_csv(Path(args.velocity_manifest))
    force_manifest = load_csv(Path(args.force_manifest))
    velocity_rows = load_csv(Path(args.velocity_csv))
    force_rows = load_csv(Path(args.force_csv))

    vel_frame_times = manifest_frame_times(velocity_manifest)
    force_frame_times = manifest_frame_times(force_manifest)

    velocity_tau_col = "instant_traversability" if "instant_traversability" in velocity_rows[0] else "supervision_traversability"
    force_tau_col = "instant_traversability" if "instant_traversability" in force_rows[0] else "supervision_traversability"

    rows_out = []
    for i, (frame_start, frame_end, label) in enumerate(DEFAULT_WINDOWS):
        vel_t0 = interpolate_timestamp(vel_frame_times, frame_start)
        vel_t1 = interpolate_timestamp(vel_frame_times, frame_end)
        force_t0 = interpolate_timestamp(force_frame_times, frame_start)
        force_t1 = interpolate_timestamp(force_frame_times, frame_end)
        include_end = i == len(DEFAULT_WINDOWS) - 1

        vel_window = select_rows(velocity_rows, vel_t0, vel_t1, include_end=include_end)
        force_window = select_rows(force_rows, force_t0, force_t1, include_end=include_end)

        vel_err = stats(row["velocity_tracking_error_filtered"] for row in vel_window)
        vel_tau = stats(row[velocity_tau_col] for row in vel_window)
        force_err = stats(row["force_error_filtered"] for row in force_window)
        force_tau = stats(row[force_tau_col] for row in force_window)

        rows_out.append(
            {
                "terrain_label": label,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "vel_samples": vel_err["n"],
                "vel_err_mean": vel_err["mean"],
                "vel_err_std": vel_err["std"],
                "vel_err_var": vel_err["var"],
                "vel_err_min": vel_err["min"],
                "vel_err_max": vel_err["max"],
                "vel_tau_mean": vel_tau["mean"],
                "vel_tau_std": vel_tau["std"],
                "vel_tau_var": vel_tau["var"],
                "vel_tau_min": vel_tau["min"],
                "vel_tau_max": vel_tau["max"],
                "force_samples": force_err["n"],
                "force_err_mean": force_err["mean"],
                "force_err_std": force_err["std"],
                "force_err_var": force_err["var"],
                "force_err_min": force_err["min"],
                "force_err_max": force_err["max"],
                "force_tau_mean": force_tau["mean"],
                "force_tau_std": force_tau["std"],
                "force_tau_var": force_tau["var"],
                "force_tau_min": force_tau["min"],
                "force_tau_max": force_tau["max"],
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[ok] wrote {output_path}")
    print("terrain|frames|vel_err_mean|vel_tau_mean|force_err_mean|force_tau_mean")
    for row in rows_out:
        print(
            f"{row['terrain_label']}|{row['frame_start']}-{row['frame_end']}|"
            f"{float(row['vel_err_mean']):.6f}|{float(row['vel_tau_mean']):.6f}|"
            f"{float(row['force_err_mean']):.6f}|{float(row['force_tau_mean']):.6f}"
        )


if __name__ == "__main__":
    main()
