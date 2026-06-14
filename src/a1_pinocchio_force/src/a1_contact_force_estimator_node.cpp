#include "a1_pinocchio_force/a1_contact_force_estimator.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "a1_contact_force_estimator");

  try {
    A1ContactForceEstimator estimator;
    ros::spin();
  } catch (const std::exception& exception) {
    ROS_FATAL("[a1_pinocchio_force] Failed to start: %s", exception.what());
    return 1;
  }

  return 0;
}
