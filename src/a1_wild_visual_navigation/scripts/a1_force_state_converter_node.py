#!/usr/bin/env python3

#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#

import message_filters
import numpy as np
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from wild_visual_navigation_msgs.msg import CustomState, RobotState


FORCE_LABELS = [
    "FLx", "FLy", "FLz",
    "FRx", "FRy", "FRz",
    "RLx", "RLy", "RLz",
    "RRx", "RRy", "RRz",
]


def make_empty_state(name, dimension):
    state = CustomState()
    state.name = name
    state.dim = dimension
    state.labels = [""] * dimension
    state.values = [0.0] * dimension
    return state


class A1ForceStateConverter:
    def __init__(self):
        self._sync_slop = rospy.get_param("~sync_slop", 0.05)
        self._transform_forces = rospy.get_param("~transform_forces", True)
        self._target_force_frame = rospy.get_param("~target_force_frame", "odom")
        self._reference_force = float(rospy.get_param("~reference_force", 134.79921))
        if self._reference_force <= 0.0:
            raise ValueError("~reference_force must be positive")
        self._transform_timeout = rospy.Duration(
            rospy.get_param("~transform_timeout", 0.05)
        )
        self._robot_state_topic = rospy.get_param(
            "~robot_state_topic", "/wild_visual_navigation_node/robot_state")
        self._odom_topic = rospy.get_param("~odom_topic", "/odom")

        foot_topics = rospy.get_param("~foot_topics", {
            "FL": "/foot_force/FL",
            "FR": "/foot_force/FR",
            "RL": "/foot_force/RL",
            "RR": "/foot_force/RR",
        })

        self._robot_state_pub = rospy.Publisher(self._robot_state_topic, RobotState, queue_size=20)
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        odom_sub = message_filters.Subscriber(self._odom_topic, Odometry)
        fl_sub = message_filters.Subscriber(foot_topics["FL"], WrenchStamped)
        fr_sub = message_filters.Subscriber(foot_topics["FR"], WrenchStamped)
        rl_sub = message_filters.Subscriber(foot_topics["RL"], WrenchStamped)
        rr_sub = message_filters.Subscriber(foot_topics["RR"], WrenchStamped)

        sync = message_filters.ApproximateTimeSynchronizer(
            [odom_sub, fl_sub, fr_sub, rl_sub, rr_sub], queue_size=50, slop=self._sync_slop)
        sync.registerCallback(self.callback)

        rospy.loginfo(
            "[a1_force_state_converter_node] ready (reference_force=%.6f N, target_force_frame=%s)",
            self._reference_force,
            self._target_force_frame,
        )

    def _transform_force(self, msg, target_frame, stamp):
        if not self._transform_forces or not msg.header.frame_id or msg.header.frame_id == target_frame:
            return msg

        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                stamp,
                self._transform_timeout,
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0,
                "[a1_force_state_converter_node] Failed to transform %s -> %s: %s",
                msg.header.frame_id,
                target_frame,
                exc,
            )
            return None

        q = transform.transform.rotation
        rotation = tf_conversions.transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

        force_vec = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z], dtype=float)
        torque_vec = np.array([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z], dtype=float)
        transformed_force = rotation.dot(force_vec)
        transformed_torque = rotation.dot(torque_vec)

        out_msg = WrenchStamped()
        out_msg.header = msg.header
        out_msg.header.frame_id = target_frame
        out_msg.header.stamp = stamp
        out_msg.wrench.force.x = float(transformed_force[0])
        out_msg.wrench.force.y = float(transformed_force[1])
        out_msg.wrench.force.z = float(transformed_force[2])
        out_msg.wrench.torque.x = float(transformed_torque[0])
        out_msg.wrench.torque.y = float(transformed_torque[1])
        out_msg.wrench.torque.z = float(transformed_torque[2])
        return out_msg

    def callback(self, odom_msg, fl_msg, fr_msg, rl_msg, rr_msg):
        target_stamp = odom_msg.header.stamp if odom_msg.header.stamp != rospy.Time() else rospy.Time(0)
        transformed_msgs = [
            self._transform_force(fl_msg, self._target_force_frame, target_stamp),
            self._transform_force(fr_msg, self._target_force_frame, target_stamp),
            self._transform_force(rl_msg, self._target_force_frame, target_stamp),
            self._transform_force(rr_msg, self._target_force_frame, target_stamp),
        ]
        if any(msg is None for msg in transformed_msgs):
            return

        fl_msg, fr_msg, rl_msg, rr_msg = transformed_msgs
        raw_values = [
            fl_msg.wrench.force.x, fl_msg.wrench.force.y, fl_msg.wrench.force.z,
            fr_msg.wrench.force.x, fr_msg.wrench.force.y, fr_msg.wrench.force.z,
            rl_msg.wrench.force.x, rl_msg.wrench.force.y, rl_msg.wrench.force.z,
            rr_msg.wrench.force.x, rr_msg.wrench.force.y, rr_msg.wrench.force.z,
        ]

        normalized_values = [float(value) / self._reference_force for value in raw_values]

        robot_state_msg = RobotState()
        robot_state_msg.header = odom_msg.header
        robot_state_msg.pose.header = odom_msg.header
        robot_state_msg.pose.pose = odom_msg.pose.pose
        robot_state_msg.twist.header = odom_msg.header
        robot_state_msg.twist.header.frame_id = odom_msg.child_frame_id
        robot_state_msg.twist.twist = odom_msg.twist.twist

        robot_state_msg.states.append(make_empty_state("joint_position", 12))
        robot_state_msg.states.append(make_empty_state("joint_velocity", 12))
        robot_state_msg.states.append(make_empty_state("joint_acceleration", 12))
        robot_state_msg.states.append(make_empty_state("joint_effort", 12))

        force_state = make_empty_state("vector_state", 12)
        force_state.labels = list(FORCE_LABELS)
        force_state.values = normalized_values
        robot_state_msg.states.append(force_state)

        self._robot_state_pub.publish(robot_state_msg)


if __name__ == "__main__":
    rospy.init_node("a1_force_state_converter_node")
    A1ForceStateConverter()
    rospy.spin()
