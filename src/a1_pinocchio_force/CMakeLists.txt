cmake_minimum_required(VERSION 3.10)
project(a1_pinocchio_force)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  message_generation
  nav_msgs
  roscpp
  sensor_msgs
  std_msgs
)

find_package(pinocchio REQUIRED)

add_message_files(
  FILES
  ContactForces.msg
  NamedArray.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs message_filters message_runtime nav_msgs roscpp sensor_msgs std_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PINOCCHIO_INCLUDE_DIRS}
)

add_executable(a1_contact_force_estimator_node
  src/a1_contact_force_estimator.cpp
  src/a1_contact_force_estimator_node.cpp
)

target_link_libraries(a1_contact_force_estimator_node
  pinocchio::pinocchio
  ${catkin_LIBRARIES}
)

add_dependencies(a1_contact_force_estimator_node
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

install(TARGETS a1_contact_force_estimator_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
