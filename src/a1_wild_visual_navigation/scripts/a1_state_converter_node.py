
#
# Copyright (c) 2022-2024, ETH Zurich, Matias Mattamala, Jonas Frey.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from wild_visual_navigation_msgs.msg import RobotState, CustomState
import rospy

# Preallocate messages
robot_state_msg = RobotState()
last_cmd_vel_msg = Twist()
has_cmd_vel = False

# Extract joint states - state 0
joint_position = CustomState()
joint_position.name = "joint_position"
joint_position.dim = 12
joint_position.labels = [""] * joint_position.dim
joint_position.values = [0] * joint_position.dim
robot_state_msg.states.append(joint_position)

# Joint velocity - state 1
joint_velocity = CustomState()
joint_velocity.name = "joint_velocity"
joint_velocity.dim = 12
joint_velocity.labels = [""] * joint_velocity.dim
joint_velocity.values = [0] * joint_velocity.dim
robot_state_msg.states.append(joint_velocity)

# Acceleration - state 2
joint_acceleration = CustomState()
joint_acceleration.name = "joint_acceleration"
joint_acceleration.dim = 12
joint_acceleration.labels = [""] * joint_acceleration.dim
joint_acceleration.values = [0] * joint_acceleration.dim
robot_state_msg.states.append(joint_acceleration)

# Effort - state 3
joint_effort = CustomState()
joint_effort.name = "joint_effort"
joint_effort.dim = 12
joint_effort.labels = [""] * joint_effort.dim
joint_effort.values = [0] * joint_effort.dim
robot_state_msg.states.append(joint_effort)

# Vector state - state 4
vector_state = CustomState()
vector_state.name = "vector_state"
vector_state.dim = 7 + 6  # + 4 * 12
vector_state.values = [0] * vector_state.dim
vector_state.labels = [""] * vector_state.dim
vector_state.values = [0] * vector_state.dim
robot_state_msg.states.append(vector_state)

i = 0
for x in ["tx", "ty", "tz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz"]:
    robot_state_msg.states[4].labels[i] = x #updating vector state
    i += 1

def a1_msg_callback(a1_state, return_msg=False):

    # For RobotState msg
    robot_state_msg.header = a1_state.header

    # Extract pose
    robot_state_msg.pose.header = a1_state.header
    robot_state_msg.pose.pose = a1_state.pose.pose

    # Extract twist
    robot_state_msg.twist.header = a1_state.header
    robot_state_msg.twist.header.frame_id = a1_state.child_frame_id
    robot_state_msg.twist.twist = a1_state.twist.twist

    robot_state_msg.states[4].values[0] = robot_state_msg.pose.pose.position.x
    robot_state_msg.states[4].values[1] = robot_state_msg.pose.pose.position.y
    robot_state_msg.states[4].values[2] = robot_state_msg.pose.pose.position.z
    robot_state_msg.states[4].values[3] = robot_state_msg.pose.pose.orientation.x
    robot_state_msg.states[4].values[4] = robot_state_msg.pose.pose.orientation.y
    robot_state_msg.states[4].values[5] = robot_state_msg.pose.pose.orientation.z
    robot_state_msg.states[4].values[6] = robot_state_msg.pose.pose.orientation.w
    robot_state_msg.states[4].values[7] = robot_state_msg.twist.twist.linear.x
    robot_state_msg.states[4].values[8] = robot_state_msg.twist.twist.linear.y
    robot_state_msg.states[4].values[9] = robot_state_msg.twist.twist.linear.z
    robot_state_msg.states[4].values[10] = robot_state_msg.twist.twist.angular.x
    robot_state_msg.states[4].values[11] = robot_state_msg.twist.twist.angular.y
    robot_state_msg.states[4].values[12] = robot_state_msg.twist.twist.angular.z

    if return_msg:
        return robot_state_msg
    # Publish
    robot_state_pub.publish(robot_state_msg)

    # Publish reference twist at odom rate for robust rosbag replay.
    out_msg = TwistStamped()
    out_msg.header.stamp = a1_state.header.stamp
    out_msg.header.frame_id = "base"
    if has_cmd_vel:
        out_msg.twist = last_cmd_vel_msg
    else:
        # Fallback when cmd_vel is absent/sparse in the bag.
        out_msg.twist = robot_state_msg.twist.twist

    x_err_ = out_msg.twist.linear.x - robot_state_msg.states[4].values[7]
    y_err_ = out_msg.twist.linear.y - robot_state_msg.states[4].values[8]
    err_msg = Float32()
    err_msg.data = (x_err_**2 + y_err_**2) / 2
    error_pub.publish(err_msg)
    ref_twiststamped_pub.publish(out_msg)


def twist_msg_callback(msg):
    global last_cmd_vel_msg
    global has_cmd_vel
    last_cmd_vel_msg = msg
    has_cmd_vel = True


if __name__ == "__main__":
    rospy.init_node("a1_state_converter_node")

    # Create publishers first so callbacks cannot fire before they exist.
    robot_state_pub = rospy.Publisher("/wild_visual_navigation_node/robot_state", RobotState, queue_size=20)
    ref_twiststamped_pub = rospy.Publisher("/wild_visual_navigation_node/reference_twist", TwistStamped, queue_size=20)
    error_pub = rospy.Publisher("/wild_visual_navigation_node/error", Float32, queue_size=20)

    # We subscribe the odometry topic (state)
    a1_state_sub = rospy.Subscriber("/odom", Odometry, a1_msg_callback, queue_size=20)
    # And also the twist command from teleoperation
    ref_twist_sub = rospy.Subscriber("/cmd_vel", Twist, twist_msg_callback, queue_size=20)

    rospy.loginfo("[a1_state_converter_node] ready")
    rospy.spin()
