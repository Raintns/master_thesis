#!/usr/bin/env python3
import rospy
import tf2_ros
import tf_conversions
import numpy as np
from geometry_msgs.msg import TransformStamped


class FootprintFromFeetBroadcaster:
    def __init__(self):
        rospy.init_node("footprint_tf_from_feet")

        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base")
        self.footprint_frame = rospy.get_param("~footprint_frame", "footprint")
        self.foot_frames = rospy.get_param("~foot_frames", ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        self.use_zero_ground_z = bool(rospy.get_param("~use_zero_ground_z", True))
        self.rate_hz = float(rospy.get_param("~rate", 50.0))

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.br = tf2_ros.TransformBroadcaster()
        self.rate = rospy.Rate(self.rate_hz)

        rospy.loginfo(
            "[footprint_tf_from_feet] Publishing %s -> %s using feet %s (z_mode=%s)",
            self.odom_frame,
            self.footprint_frame,
            ", ".join(self.foot_frames),
            "zero" if self.use_zero_ground_z else "feet_average",
        )

    def run(self):
        while not rospy.is_shutdown():
            try:
                foot_positions = []
                for frame in self.foot_frames:
                    t = self.tf_buffer.lookup_transform(
                        self.odom_frame,
                        frame,
                        rospy.Time(0),
                        timeout=rospy.Duration(0.1),
                    )
                    foot_positions.append(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                        ]
                    )

                foot_array = np.array(foot_positions, dtype=np.float64)
                center_x, center_y, center_z = np.mean(foot_array, axis=0)

                base_tf = self.tf_buffer.lookup_transform(
                    self.odom_frame,
                    self.base_frame,
                    rospy.Time(0),
                    timeout=rospy.Duration(0.1),
                )
                q = base_tf.transform.rotation
                _, _, yaw = tf_conversions.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
                flat_q = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, yaw)

                footprint = TransformStamped()
                footprint.header.stamp = rospy.Time.now()
                footprint.header.frame_id = self.odom_frame
                footprint.child_frame_id = self.footprint_frame
                footprint.transform.translation.x = float(center_x)
                footprint.transform.translation.y = float(center_y)
                footprint.transform.translation.z = 0.0 if self.use_zero_ground_z else float(center_z)
                footprint.transform.rotation.x = flat_q[0]
                footprint.transform.rotation.y = flat_q[1]
                footprint.transform.rotation.z = flat_q[2]
                footprint.transform.rotation.w = flat_q[3]
                self.br.sendTransform(footprint)

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                pass

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = FootprintFromFeetBroadcaster()
        node.run()
    except rospy.ROSInterruptException:
        pass
