#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rosbag


DEFAULT_BAG = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_velocity_debug.bag"
)
DEFAULT_MANIFEST = Path(
    "/home/rain/github_upload/Result/velocity_default(with_frame_output_detail)/wvn_frames/manifest.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/rain/github_upload/Result/rendered_figures/sparse_command_vs_actual_velocity.png"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot sparse teleop command velocity against the synchronized WVN reference "
            "and measured robot velocity."
        )
    )
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--reference-topic", default="/wild_visual_navigation_node/reference_twist")
    parser.add_argument("--state-topic", default="/wild_visual_navigation_node/robot_state")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--range-start", type=int, default=None)
    parser.add_argument("--range-end", type=int, default=None)
    return parser.parse_args()


def load_frame_window(manifest_path, range_start, range_end):
    if manifest_path is None or range_start is None or range_end is None:
        return None

    start_time = None
    end_time = None
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            frame_index = int(row["frame_index"])
            if frame_index == range_start:
                start_time = float(row["stamp_secs"]) + float(row["stamp_nsecs"]) * 1e-9
            if frame_index == range_end:
                end_time = float(row["stamp_secs"]) + float(row["stamp_nsecs"]) * 1e-9

    if start_time is None or end_time is None:
        raise KeyError(
            f"Could not find frame window [{range_start}, {range_end}] in {manifest_path}"
        )
    return start_time, end_time


def append_series(series, timestamp, vx, wz):
    series["t"].append(timestamp)
    series["vx"].append(vx)
    series["wz"].append(wz)


def read_velocity_series(bag_path, cmd_topic, reference_topic, state_topic, time_window=None):
    topics = [cmd_topic, reference_topic, state_topic]
    cmd = {"t": [], "vx": [], "wz": []}
    reference = {"t": [], "vx": [], "wz": []}
    measured = {"t": [], "vx": [], "wz": []}

    with rosbag.Bag(str(bag_path), "r") as bag:
        first_time = None
        for topic, msg, bag_time in bag.read_messages(topics=topics):
            timestamp = bag_time.to_sec()
            if time_window is not None:
                if timestamp < time_window[0] or timestamp > time_window[1]:
                    continue

            if first_time is None:
                first_time = timestamp

            relative_time = timestamp - first_time
            if topic == cmd_topic:
                append_series(cmd, relative_time, msg.linear.x, msg.angular.z)
            elif topic == reference_topic:
                append_series(reference, relative_time, msg.twist.linear.x, msg.twist.angular.z)
            elif topic == state_topic:
                append_series(measured, relative_time, msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    return cmd, reference, measured


def plot_panel(ax, cmd, reference, measured, component_key, ylabel, title):
    ax.step(
        cmd["t"],
        cmd[component_key],
        where="post",
        color="#d62728",
        linewidth=1.3,
        alpha=0.95,
        label="Raw teleop command (/cmd_vel)",
    )
    ax.scatter(
        cmd["t"],
        cmd[component_key],
        color="#d62728",
        s=16,
        alpha=0.9,
        zorder=3,
    )
    ax.plot(
        reference["t"],
        reference[component_key],
        color="#ff7f0e",
        linewidth=1.5,
        linestyle="--",
        label="Synchronized WVN reference",
    )
    ax.plot(
        measured["t"],
        measured[component_key],
        color="#1f77b4",
        linewidth=1.5,
        label="Measured robot velocity",
    )
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    ax.minorticks_on()


def render_figure(cmd, reference, measured, output_path, title_suffix=""):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 8.5),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    heading = "Sparse Teleoperation Commands vs Synchronized and Measured Velocity"
    if title_suffix:
        heading = f"{heading} ({title_suffix})"
    fig.suptitle(heading, fontsize=18)

    plot_panel(
        axes[0],
        cmd=cmd,
        reference=reference,
        measured=measured,
        component_key="vx",
        ylabel="Linear velocity $v_x$ [m/s]",
        title="Linear Velocity Comparison",
    )
    plot_panel(
        axes[1],
        cmd=cmd,
        reference=reference,
        measured=measured,
        component_key="wz",
        ylabel="Yaw rate $\\omega_z$ [rad/s]",
        title="Angular Velocity Comparison",
    )
    axes[1].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right", frameon=True)

    summary = (
        f"Raw /cmd_vel: {len(cmd['t'])} msgs    "
        f"reference_twist: {len(reference['t'])} msgs    "
        f"robot_state: {len(measured['t'])} msgs"
    )
    fig.text(0.5, 0.955, summary, ha="center", va="top", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    if not args.bag.is_file():
        raise FileNotFoundError(f"Bag not found: {args.bag}")

    time_window = load_frame_window(args.manifest, args.range_start, args.range_end)
    cmd, reference, measured = read_velocity_series(
        bag_path=args.bag,
        cmd_topic=args.cmd_topic,
        reference_topic=args.reference_topic,
        state_topic=args.state_topic,
        time_window=time_window,
    )

    suffix = ""
    if args.range_start is not None and args.range_end is not None:
        suffix = f"frames {args.range_start}-{args.range_end}"

    render_figure(cmd, reference, measured, args.output, title_suffix=suffix)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
