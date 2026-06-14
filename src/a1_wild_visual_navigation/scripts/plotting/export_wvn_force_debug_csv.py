#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys

import rosbag


DEFAULT_TOPICS = [
    "/wild_visual_navigation_node/debug/current_force",
    "/wild_visual_navigation_node/debug/desired_force",
    "/wild_visual_navigation_node/debug/force_error",
    "/wild_visual_navigation_node/debug/force_error_mse",
    "/wild_visual_navigation_node/debug/force_error_filtered",
    "/wild_visual_navigation_node/debug/raw_current_force",
    "/wild_visual_navigation_node/debug/raw_desired_force",
    "/wild_visual_navigation_node/debug/raw_force_error",
    "/wild_visual_navigation_node/debug/raw_force_error_mse",
    "/wild_visual_navigation_node/debug/raw_force_error_rmse",
]


def sanitize_topic_name(topic_name):
    return re.sub(r"[^A-Za-z0-9]+", "_", topic_name).strip("_")


def export_custom_state(topic_name, messages, output_dir):
    rows = []
    labels = None

    for _, msg, bag_time in messages:
        if labels is None:
            labels = list(msg.labels)
        row = {
            "timestamp": msg.header.stamp.to_sec() if hasattr(msg, "header") else bag_time.to_sec(),
            "name": msg.name,
        }
        for label, value in zip(msg.labels, msg.values):
            row[label] = value
        rows.append(row)

    if not rows:
        return None

    csv_path = os.path.join(output_dir, f"{sanitize_topic_name(topic_name)}.csv")
    fieldnames = ["timestamp", "name"] + list(labels or [])
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def export_float32(topic_name, messages, output_dir):
    csv_path = os.path.join(output_dir, f"{sanitize_topic_name(topic_name)}.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", "data"])
        for _, msg, bag_time in messages:
            writer.writerow([bag_time.to_sec(), msg.data])
    return csv_path


def collect_topic_messages(bag_path, topic_name):
    with rosbag.Bag(bag_path, "r") as bag:
        return list(bag.read_messages(topics=[topic_name]))


def main():
    parser = argparse.ArgumentParser(
        description="Export WVN force debug topics from a rosbag into CSV files."
    )
    parser.add_argument("bag", help="Input rosbag that already contains the debug topics.")
    parser.add_argument(
        "--output-dir",
        default="force_debug_csv",
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_TOPICS,
        help="Topics to export. Defaults to the WVN force debug topics.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bag):
        print(f"Bag file not found: {args.bag}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)

    exported_any = False
    for topic_name in args.topics:
        messages = collect_topic_messages(args.bag, topic_name)
        if not messages:
            print(f"[skip] no messages found for {topic_name}")
            continue

        first_msg = messages[0][1]
        if hasattr(first_msg, "labels") and hasattr(first_msg, "values"):
            csv_path = export_custom_state(topic_name, messages, args.output_dir)
        elif hasattr(first_msg, "data"):
            csv_path = export_float32(topic_name, messages, args.output_dir)
        else:
            print(f"[skip] unsupported message type on {topic_name}: {type(first_msg).__name__}")
            continue

        exported_any = True
        print(f"[ok] wrote {csv_path}")

    if not exported_any:
        print(
            "No CSV files were written. Make sure the bag actually contains the debug topics.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
