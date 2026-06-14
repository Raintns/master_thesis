#!/usr/bin/env python3

import csv
from pathlib import Path

import cv2
import message_filters
from matplotlib import cm
import numpy as np
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


OPENCV_COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def parse_string_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()]


def parse_frame_indices(value):
    items = parse_string_list(value)
    return {int(item) for item in items}


def stamp_to_string(stamp):
    return f"{stamp.secs:010d}_{stamp.nsecs:09d}"


class ComparisonFrameSaver:
    def __init__(self):
        self._bridge = CvBridge()
        self._read_params()
        self._setup_output()
        self._dump_parameter_snapshots()
        self._setup_manifest()
        self._setup_subscribers()

        rospy.loginfo(
            "[wvn_comparison_frame_saver] Saving comparison frames to %s",
            self._output_dir,
        )

    def _read_params(self):
        self._output_dir = Path(rospy.get_param("~output_dir", "/tmp/wvn_comparison"))
        self._raw_topic = rospy.get_param(
            "~raw_topic",
            "/wild_visual_navigation_node/front/image_input",
        )
        self._trav_topic = rospy.get_param(
            "~trav_topic",
            "/wild_visual_navigation_node/front/traversability",
        )
        self._confidence_topic = rospy.get_param(
            "~confidence_topic",
            "/wild_visual_navigation_node/front/confidence",
        )
        self._queue_size = int(rospy.get_param("~queue_size", 10))
        self._slop = float(rospy.get_param("~slop", 0.2))
        self._save_every_n = max(1, int(rospy.get_param("~save_every_n", 30)))
        self._start_frame = max(0, int(rospy.get_param("~start_frame", 0)))
        self._max_saved_frames = max(0, int(rospy.get_param("~max_saved_frames", 0)))
        self._frame_indices = parse_frame_indices(rospy.get_param("~frame_indices", ""))
        self._save_prediction_arrays = bool(rospy.get_param("~save_prediction_arrays", True))
        self._overlay_alpha = float(rospy.get_param("~overlay_alpha", 0.45))
        self._value_min = float(rospy.get_param("~value_min", 0.0))
        self._value_max = float(rospy.get_param("~value_max", 1.0))
        self._normalize_by_minmax = bool(rospy.get_param("~normalize_by_minmax", False))
        self._param_namespaces = parse_string_list(
            rospy.get_param(
                "~param_namespaces",
                ["/wvn_learning_node", "/wvn_feature_extractor_node"],
            )
        )
        self._scalar_topic_max_age = float(rospy.get_param("~scalar_topic_max_age", 0.2))
        self._instant_traversability_topic = rospy.get_param(
            "~instant_traversability_topic",
            "/wild_visual_navigation_node/instant_traversability",
        )
        self._velocity_error_topic = rospy.get_param(
            "~velocity_error_topic",
            "/wild_visual_navigation_node/debug/velocity_tracking_error_mse",
        )
        self._velocity_error_filtered_topic = rospy.get_param(
            "~velocity_error_filtered_topic",
            "/wild_visual_navigation_node/debug/velocity_tracking_error_filtered",
        )
        self._force_error_topic = rospy.get_param(
            "~force_error_topic",
            "/wild_visual_navigation_node/debug/force_error_mse",
        )
        self._force_error_filtered_topic = rospy.get_param(
            "~force_error_filtered_topic",
            "/wild_visual_navigation_node/debug/force_error_filtered",
        )
        self._scalar_topic_specs = [
            ("instant_traversability", self._instant_traversability_topic),
            ("velocity_error_mse", self._velocity_error_topic),
            ("velocity_error_filtered", self._velocity_error_filtered_topic),
            ("force_error_mse", self._force_error_topic),
            ("force_error_filtered", self._force_error_filtered_topic),
        ]
        self._latest_scalar_values = {}

        self._colormap_name, self._colormap_backend, self._colormap = self._resolve_colormap(
            rospy.get_param("~colormap", "RdYlBu")
        )

        self._seen_frames = 0
        self._saved_frames = 0

    def _setup_output(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _dump_parameter_snapshots(self):
        snapshot = {
            "comparison_frame_saver": {
                "raw_topic": self._raw_topic,
                "trav_topic": self._trav_topic,
                "confidence_topic": self._confidence_topic,
                "frame_indices": sorted(self._frame_indices),
                "save_every_n": self._save_every_n,
                "start_frame": self._start_frame,
                "max_saved_frames": self._max_saved_frames,
                "overlay_alpha": self._overlay_alpha,
                "colormap": self._colormap_name,
                "value_min": self._value_min,
                "value_max": self._value_max,
                "normalize_by_minmax": self._normalize_by_minmax,
                "scalar_topic_max_age": self._scalar_topic_max_age,
                "instant_traversability_topic": self._instant_traversability_topic,
                "velocity_error_topic": self._velocity_error_topic,
                "velocity_error_filtered_topic": self._velocity_error_filtered_topic,
                "force_error_topic": self._force_error_topic,
                "force_error_filtered_topic": self._force_error_filtered_topic,
            }
        }

        for namespace in self._param_namespaces:
            try:
                snapshot[namespace] = rospy.get_param(namespace)
            except KeyError:
                rospy.logwarn(
                    "[wvn_comparison_frame_saver] Could not read ROS params under %s",
                    namespace,
                )

        with open(self._output_dir / "params_snapshot.yaml", "w") as handle:
            yaml.safe_dump(snapshot, handle, sort_keys=False)

    def _setup_manifest(self):
        manifest_path = self._output_dir / "manifest.csv"
        self._manifest_file = open(manifest_path, "w", newline="")
        self._manifest_writer = csv.writer(self._manifest_file)
        self._manifest_writer.writerow(
            [
                "frame_index",
                "stamp_secs",
                "stamp_nsecs",
                "frame_dir",
                "raw_topic",
                "trav_topic",
                "confidence_topic",
                "instant_traversability",
                "velocity_error_mse",
                "velocity_error_filtered",
                "force_error_mse",
                "force_error_filtered",
            ]
        )
        self._manifest_file.flush()

    def _setup_subscribers(self):
        subscribers = [
            message_filters.Subscriber(self._raw_topic, Image),
            message_filters.Subscriber(self._trav_topic, Image),
        ]
        if self._confidence_topic:
            subscribers.append(message_filters.Subscriber(self._confidence_topic, Image))

        sync = message_filters.ApproximateTimeSynchronizer(
            subscribers,
            queue_size=self._queue_size,
            slop=self._slop,
        )
        sync.registerCallback(self._callback)
        self._sync = sync

        self._scalar_subscribers = []
        for key, topic in self._scalar_topic_specs:
            if not topic:
                continue
            subscriber = rospy.Subscriber(
                topic,
                Float32,
                self._scalar_callback,
                callback_args=key,
                queue_size=20,
            )
            self._scalar_subscribers.append(subscriber)

    def _scalar_callback(self, msg, key):
        self._latest_scalar_values[key] = {
            "stamp": rospy.Time.now(),
            "value": float(msg.data),
        }

    def _resolve_colormap(self, name):
        name = str(name).strip()
        if not name:
            name = "RdYlBu"

        opencv_key = name.lower()
        if opencv_key in OPENCV_COLORMAPS:
            return opencv_key, "opencv", OPENCV_COLORMAPS[opencv_key]

        try:
            return name, "matplotlib", cm.get_cmap(name)
        except ValueError:
            rospy.logwarn(
                "[wvn_comparison_frame_saver] Unknown colormap '%s', using RdYlBu",
                name,
            )
            return "RdYlBu", "matplotlib", cm.get_cmap("RdYlBu")

    def _should_save(self, frame_index):
        if frame_index < self._start_frame:
            return False
        if self._frame_indices:
            return frame_index in self._frame_indices
        return (frame_index - self._start_frame) % self._save_every_n == 0

    def _image_to_bgr(self, image_msg):
        encoding = (image_msg.encoding or "").lower()

        if encoding.startswith("rgb"):
            rgb = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="rgb8")
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if encoding.startswith("bgr"):
            return self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        image = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        image = np.asarray(image)

        if image.ndim == 2:
            mono = self._to_uint8(image)
            return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.ndim == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                return self._to_uint8(image)
            return image

        raise ValueError(f"Unsupported image shape {image.shape} for topic {self._raw_topic}")

    def _value_to_float(self, image_msg):
        value = self._bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 3:
            value = value[:, :, 0]
        value = np.nan_to_num(value, nan=0.0, posinf=self._value_max, neginf=self._value_min)

        if self._normalize_by_minmax:
            min_value = float(np.min(value))
            max_value = float(np.max(value))
            if max_value > min_value:
                normalized = (value - min_value) / (max_value - min_value)
            else:
                normalized = np.zeros_like(value)
        else:
            clipped = np.clip(value, self._value_min, self._value_max)
            if self._value_max > self._value_min:
                normalized = (clipped - self._value_min) / (self._value_max - self._value_min)
            else:
                normalized = np.zeros_like(clipped)

        return value, np.clip(normalized, 0.0, 1.0)

    def _to_uint8(self, image):
        image = np.asarray(image)
        if image.dtype == np.uint8:
            return image

        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        min_value = float(np.min(image))
        max_value = float(np.max(image))
        if max_value <= min_value:
            return np.zeros_like(image, dtype=np.uint8)
        scaled = (image - min_value) / (max_value - min_value)
        return np.uint8(np.round(scaled * 255.0))

    def _resize_to_match(self, image, reference):
        if image.shape[:2] == reference.shape[:2]:
            return image
        return cv2.resize(
            image,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    def _colorize(self, normalized):
        if self._colormap_backend == "opencv":
            value_u8 = np.uint8(np.round(normalized * 255.0))
            return cv2.applyColorMap(value_u8, self._colormap)

        rgb = np.uint8(np.round(self._colormap(normalized)[..., :3] * 255.0))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _save_value_products(self, frame_dir, stem, value_msg, raw_bgr):
        raw_value, normalized = self._value_to_float(value_msg)

        if self._save_prediction_arrays:
            np.save(frame_dir / f"{stem}.npy", raw_value)

        colorized = self._colorize(normalized)
        colorized = self._resize_to_match(colorized, raw_bgr)
        overlay = cv2.addWeighted(raw_bgr, 1.0 - self._overlay_alpha, colorized, self._overlay_alpha, 0.0)

        cv2.imwrite(str(frame_dir / f"{stem}_color.png"), colorized)
        cv2.imwrite(str(frame_dir / f"{stem}_overlay.png"), overlay)

    def _get_scalar_metadata(self):
        metadata = {}
        now = rospy.Time.now()
        for key, _topic in self._scalar_topic_specs:
            entry = self._latest_scalar_values.get(key)
            if entry is None:
                metadata[key] = None
                continue
            age = abs((now - entry["stamp"]).to_sec())
            metadata[key] = entry["value"] if age <= self._scalar_topic_max_age else None
        return metadata

    def _callback(self, *msgs):
        frame_index = self._seen_frames
        self._seen_frames += 1

        if not self._should_save(frame_index):
            return

        if self._max_saved_frames and self._saved_frames >= self._max_saved_frames:
            return

        raw_msg = msgs[0]
        trav_msg = msgs[1]
        confidence_msg = msgs[2] if len(msgs) > 2 else None

        stamp = raw_msg.header.stamp if raw_msg.header.stamp != rospy.Time() else trav_msg.header.stamp
        frame_dir = self._output_dir / f"frame_{frame_index:06d}_{stamp_to_string(stamp)}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        raw_bgr = self._image_to_bgr(raw_msg)
        cv2.imwrite(str(frame_dir / "raw.png"), raw_bgr)
        self._save_value_products(frame_dir, "traversability", trav_msg, raw_bgr)

        if confidence_msg is not None:
            self._save_value_products(frame_dir, "confidence", confidence_msg, raw_bgr)

        scalar_metadata = self._get_scalar_metadata()
        metadata = {
            "frame_index": int(frame_index),
            "stamp_secs": int(stamp.secs),
            "stamp_nsecs": int(stamp.nsecs),
            "frame_dir": frame_dir.name,
            "raw_topic": self._raw_topic,
            "trav_topic": self._trav_topic,
            "confidence_topic": self._confidence_topic,
            **scalar_metadata,
        }
        with open(frame_dir / "metadata.yaml", "w") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False)

        self._manifest_writer.writerow(
            [
                frame_index,
                stamp.secs,
                stamp.nsecs,
                frame_dir.name,
                self._raw_topic,
                self._trav_topic,
                self._confidence_topic,
                scalar_metadata["instant_traversability"],
                scalar_metadata["velocity_error_mse"],
                scalar_metadata["velocity_error_filtered"],
                scalar_metadata["force_error_mse"],
                scalar_metadata["force_error_filtered"],
            ]
        )
        self._manifest_file.flush()

        self._saved_frames += 1
        rospy.loginfo(
            "[wvn_comparison_frame_saver] Saved frame %d at %s",
            frame_index,
            frame_dir,
        )

        if self._max_saved_frames and self._saved_frames >= self._max_saved_frames:
            rospy.signal_shutdown("Requested frame set has been saved")


if __name__ == "__main__":
    rospy.init_node("wvn_comparison_frame_saver")
    saver = ComparisonFrameSaver()
    rospy.spin()
