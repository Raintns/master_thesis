#!/usr/bin/env python3

#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#

import rospy
from geometry_msgs.msg import WrenchStamped

from a1_pinocchio_force.msg import ContactForces
from aliengo_dynamics_computer.msg import ReactionForce


TARGET_ORDER = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


def canonical_label(label):
    label_lower = label.lower()
    if label_lower in ("fl", "fl_foot"):
        return "FL_foot"
    if label_lower in ("fr", "fr_foot"):
        return "FR_foot"
    if label_lower in ("rl", "rl_foot"):
        return "RL_foot"
    if label_lower in ("rr", "rr_foot"):
        return "RR_foot"
    return ""


class A1ForceReferenceAdapter:
    def __init__(self):
        self._input_topic = rospy.get_param(
            "~input_topic", "/a1_pinocchio_force/estimated_contact_forces")
        self._output_topic = rospy.get_param(
            "~output_topic", "/a1_force_wvn/reference_reaction_force")
        self._reference_force = float(rospy.get_param("~reference_force", 134.79921))
        if self._reference_force <= 0.0:
            raise ValueError("~reference_force must be positive")

        self._publisher = rospy.Publisher(self._output_topic, ReactionForce, queue_size=20)
        self._subscriber = rospy.Subscriber(
            self._input_topic, ContactForces, self.callback, queue_size=20)

        rospy.loginfo(
            "[a1_force_reference_adapter_node] ready (reference_force=%.6f N)",
            self._reference_force,
        )

    def callback(self, msg):
        force_by_label = {}
        for label, force_msg in zip(msg.labels, msg.forces):
            normalized_label = canonical_label(label)
            if normalized_label:
                force_by_label[normalized_label] = force_msg

        output_msg = ReactionForce()
        output_msg.header = msg.header

        scaled_values = []
        for label in TARGET_ORDER:
            force_msg = force_by_label.get(label, WrenchStamped())
            force_msg.header = msg.header
            scaled_values.extend([
                float(force_msg.wrench.force.x) / self._reference_force,
                float(force_msg.wrench.force.y) / self._reference_force,
                float(force_msg.wrench.force.z) / self._reference_force,
            ])

        for index, label in enumerate(TARGET_ORDER):
            force_msg = WrenchStamped()
            force_msg.header = msg.header
            force_msg.wrench.force.x = scaled_values[3 * index]
            force_msg.wrench.force.y = scaled_values[3 * index + 1]
            force_msg.wrench.force.z = scaled_values[3 * index + 2]
            output_msg.reaction_forces.append(force_msg)

        self._publisher.publish(output_msg)


if __name__ == "__main__":
    rospy.init_node("a1_force_reference_adapter_node")
    A1ForceReferenceAdapter()
    rospy.spin()
