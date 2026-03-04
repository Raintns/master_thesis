#!/usr/bin/env python3
import rospy
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped


def main():
    rospy.init_node("footprint_tf_publisher")

    parent_frame = rospy.get_param("~parent_frame", "odom")
    base_frame = rospy.get_param("~base_frame", "base")
    footprint_frame = rospy.get_param("~footprint_frame", "footprint")
    ground_z = float(rospy.get_param("~ground_z", 0.0))
    rate_hz = float(rospy.get_param("~rate", 50.0))

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    tf2_ros.TransformListener(tf_buffer)
    br = tf2_ros.TransformBroadcaster()
    rate = rospy.Rate(rate_hz)

    rospy.loginfo(
        "[footprint_tf_publisher] Publishing dynamic TF %s -> %s projected from %s (z=%.3f, rate=%.1f Hz)",
        parent_frame,
        footprint_frame,
        base_frame,
        ground_z,
        rate_hz,
    )

    while not rospy.is_shutdown():
        try:
            tf_base = tf_buffer.lookup_transform(
                parent_frame,
                base_frame,
                rospy.Time(0),
                timeout=rospy.Duration(0.2),
            )
            q = tf_base.transform.rotation
            _, _, yaw = tf_conversions.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
            q_yaw = tf_conversions.transformations.quaternion_from_euler(0.0, 0.0, yaw)

            msg = TransformStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = parent_frame
            msg.child_frame_id = footprint_frame
            msg.transform.translation.x = tf_base.transform.translation.x
            msg.transform.translation.y = tf_base.transform.translation.y
            msg.transform.translation.z = ground_z
            msg.transform.rotation.x = q_yaw[0]
            msg.transform.rotation.y = q_yaw[1]
            msg.transform.rotation.z = q_yaw[2]
            msg.transform.rotation.w = q_yaw[3]
            br.sendTransform(msg)
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(
                5.0,
                "[footprint_tf_publisher] Waiting for TF %s -> %s (%s)",
                parent_frame,
                base_frame,
                str(e),
            )
        rate.sleep()


if __name__ == "__main__":
    main()
