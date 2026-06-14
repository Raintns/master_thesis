#!/usr/bin/env python3

import argparse
import csv
import os
import sys

import rosbag


TOPIC_SPECS = [
    ("/wild_visual_navigation_node/instant_traversability", "instant_traversability", "float32"),
    (
        "/wild_visual_navigation_node/debug/supervision_traversability",
        "supervision_traversability",
        "float32",
    ),
    (
        "/wild_visual_navigation_node/debug/supervision_traversability_var",
        "supervision_traversability_var",
        "float32",
    ),
    (
        "/wild_visual_navigation_node/debug/supervision_is_untraversable",
        "supervision_is_untraversable",
        "float32",
    ),
    (
        "/wild_visual_navigation_node/debug/force_supervision_traversability",
        "force_supervision_traversability",
        "float32",
    ),
    (
        "/wild_visual_navigation_node/debug/velocity_supervision_traversability",
        "velocity_supervision_traversability",
        "float32",
    ),
    ("/wild_visual_navigation_node/debug/current_force", "current_force", "custom_state"),
    ("/wild_visual_navigation_node/debug/desired_force", "desired_force", "custom_state"),
    ("/wild_visual_navigation_node/debug/force_error", "force_error", "custom_state"),
    ("/wild_visual_navigation_node/debug/force_error_mse", "force_error_mse", "float32"),
    ("/wild_visual_navigation_node/debug/force_error_filtered", "force_error_filtered", "float32"),
    ("/wild_visual_navigation_node/debug/raw_current_force", "raw_current_force", "custom_state"),
    ("/wild_visual_navigation_node/debug/raw_desired_force", "raw_desired_force", "custom_state"),
    ("/wild_visual_navigation_node/debug/raw_force_error", "raw_force_error", "custom_state"),
    ("/wild_visual_navigation_node/debug/raw_force_error_mse", "raw_force_error_mse", "float32"),
    ("/wild_visual_navigation_node/debug/raw_force_error_rmse", "raw_force_error_rmse", "float32"),
    ("/wild_visual_navigation_node/debug/current_twist", "current_twist", "custom_state"),
    ("/wild_visual_navigation_node/debug/desired_twist", "desired_twist", "custom_state"),
    ("/wild_visual_navigation_node/debug/twist_error", "twist_error", "custom_state"),
    ("/wild_visual_navigation_node/debug/velocity_tracking_error_mse", "velocity_tracking_error_mse", "float32"),
    (
        "/wild_visual_navigation_node/debug/velocity_tracking_error_filtered",
        "velocity_tracking_error_filtered",
        "float32",
    ),
    (
        "/wild_visual_navigation_node/debug/confidence_generator_mean",
        "confidence_generator_mean",
        "float32",
    ),
    (
        "/wild_visual_navigation_node/debug/confidence_generator_std",
        "confidence_generator_std",
        "float32",
    ),
    (
        "/wild_visual_navigation_node/debug/confidence_generator_var",
        "confidence_generator_var",
        "float32",
    ),
]


def read_messages(bag_path, topics):
    with rosbag.Bag(bag_path, "r") as bag:
        return list(bag.read_messages(topics=topics))


def find_or_create_row(rows, timestamp, tolerance, source_bag):
    if rows and abs(rows[-1]["timestamp"] - timestamp) <= tolerance:
        return rows[-1]

    row = {"timestamp": timestamp, "source_bag": source_bag}
    rows.append(row)
    return row


def update_custom_state_row(row, prefix, msg, ordered_columns):
    for label, value in zip(msg.labels, msg.values):
        column = f"{prefix}_{label}"
        row[column] = value
        if column not in ordered_columns:
            ordered_columns.append(column)


def update_float_row(row, column, msg, ordered_columns):
    row[column] = msg.data
    if column not in ordered_columns:
        ordered_columns.append(column)


def export_debug_csv(bag_paths, output_csv, tolerance):
    topics = [topic for topic, _, _ in TOPIC_SPECS]
    topic_to_spec = {topic: (prefix, msg_type) for topic, prefix, msg_type in TOPIC_SPECS}

    rows = []
    ordered_columns = []
    exported_any = False
    for bag_path in bag_paths:
        messages = read_messages(bag_path, topics)
        if not messages:
            continue

        exported_any = True
        source_bag = os.path.basename(bag_path)
        bag_rows = []
        for topic, msg, bag_time in messages:
            timestamp = bag_time.to_sec()
            row = find_or_create_row(bag_rows, timestamp, tolerance, source_bag)
            prefix, msg_type = topic_to_spec[topic]

            if msg_type == "custom_state":
                update_custom_state_row(row, prefix, msg, ordered_columns)
            elif msg_type == "float32":
                update_float_row(row, prefix, msg, ordered_columns)

        rows.extend(bag_rows)

    if not exported_any:
        return False, []

    fieldnames = ["timestamp", "source_bag"] + ordered_columns
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return True, fieldnames


def main():
    parser = argparse.ArgumentParser(
        description="Export WVN force and velocity debug topics from a rosbag into one merged CSV file."
    )
    parser.add_argument("bags", nargs="+", help="One or more input rosbags containing WVN debug topics.")
    parser.add_argument(
        "--output",
        default="wvn_debug.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="Timestamp grouping tolerance in seconds for merging topic samples into one row.",
    )
    args = parser.parse_args()

    missing_bags = [bag_path for bag_path in args.bags if not os.path.isfile(bag_path)]
    if missing_bags:
        for bag_path in missing_bags:
            print(f"Bag file not found: {bag_path}", file=sys.stderr)
        return 1

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    exported, fieldnames = export_debug_csv(args.bags, args.output, args.tolerance)
    if not exported:
        print(
            "No WVN debug topics were found in the bag. Replay and record the debug topics first.",
            file=sys.stderr,
        )
        return 2

    print(f"[ok] wrote {args.output}")
    print(f"[ok] columns: {', '.join(fieldnames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
