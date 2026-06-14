cmake_minimum_required(VERSION 3.0.2)
project(a1_wild_visual_navigation)

find_package(catkin REQUIRED COMPONENTS
  a1_pinocchio_force
  aliengo_dynamics_computer
  geometry_msgs
  message_filters
  rospy
  nav_msgs
  sensor_msgs
  std_msgs
  wild_visual_navigation_msgs
  wild_visual_navigation_ros
)

catkin_package(
  CATKIN_DEPENDS
)

message(${CATKIN_PACKAGE_SHARE_DESTINATION})
install(DIRECTORY launch Media worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

############
## Python ##
############
catkin_install_python(PROGRAMS scripts/carrot_follower.py
                               scripts/a1_state_converter_node.py
                               scripts/a1_force_state_converter_node.py
                               scripts/a1_force_reference_adapter_node.py
                               scripts/export_wvn_force_debug_csv.py
                               scripts/export_wvn_debug_csv.py
                               scripts/gazebo_world_publisher.py
                               scripts/footprint_tf_publisher.py
                               scripts/footprint_tf_from_feet.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
