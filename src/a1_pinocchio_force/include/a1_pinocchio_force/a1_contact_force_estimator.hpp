#ifndef A1_PINOCCHIO_FORCE_A1_CONTACT_FORCE_ESTIMATOR_HPP
#define A1_PINOCCHIO_FORCE_A1_CONTACT_FORCE_ESTIMATOR_HPP

#include <memory>
#include <string>
#include <vector>

// Include Pinocchio headers first to avoid ROS/Boost compilation issues.
#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/joint/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <Eigen/Dense>

#include <geometry_msgs/WrenchStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>

#include "a1_pinocchio_force/ContactForces.h"
#include "a1_pinocchio_force/NamedArray.h"

class A1ContactForceEstimator
{
public:
  A1ContactForceEstimator();

private:
  using StateSyncPolicy =
    message_filters::sync_policies::ApproximateTime<
      sensor_msgs::JointState,
      nav_msgs::Odometry,
      sensor_msgs::Imu>;

  struct ScalarKalmanFilter
  {
    bool initialized = false;
    double estimate = 0.0;
    double covariance = 1.0;
    double process_noise = 0.5;
    double measurement_noise = 4.0;

    void configure(double process, double measurement, double initial_covariance);
    double update(double measurement);
    void reset();
  };

  struct ContactTopicState
  {
    bool received = false;
    bool active = true;
    ros::Time stamp;
    double norm = 0.0;
  };

  void loadParams();
  void buildModel();
  void initializeActuatedJointNames();

  void stateCallback(
    const sensor_msgs::JointState::ConstPtr& joint_state_msg,
    const nav_msgs::Odometry::ConstPtr& odom_msg,
    const sensor_msgs::Imu::ConstPtr& imu_msg);
  void contactCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg, std::size_t index);

  bool buildJointVectors(
    const sensor_msgs::JointState& msg,
    Eigen::VectorXd& joint_position,
    Eigen::VectorXd& joint_velocity,
    Eigen::VectorXd& joint_torque) const;

  std::vector<std::size_t> activeContactIndices(const ros::Time& stamp) const;
  Eigen::VectorXd filterJointAcceleration(const Eigen::VectorXd& raw_joint_acceleration);
  Eigen::VectorXd filterBaseAcceleration(const Eigen::VectorXd& raw_base_acceleration);
  Eigen::Vector3d gravityVectorInBodyFrame(const Eigen::Quaterniond& orientation) const;
  void publishResults(
    const ros::Time& stamp,
    const std::string& output_frame,
    const std::vector<Eigen::Vector3d>& contact_forces,
    const std::vector<bool>& contact_mask);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  std::unique_ptr<message_filters::Subscriber<sensor_msgs::JointState>> joint_state_sub_;
  std::unique_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  std::unique_ptr<message_filters::Subscriber<sensor_msgs::Imu>> imu_sub_;
  std::unique_ptr<message_filters::Synchronizer<StateSyncPolicy>> state_sync_;
  std::vector<ros::Subscriber> contact_subs_;

  ros::Publisher contact_forces_pub_;
  ros::Publisher force_magnitudes_pub_;
  ros::Publisher active_contact_mask_pub_;
  std::vector<ros::Publisher> per_foot_force_pubs_;

  std::string robot_urdf_;
  std::string joint_states_topic_;
  std::string odom_topic_;
  std::string imu_topic_;
  std::string output_frame_;

  bool use_contact_topics_ = true;
  bool publish_per_foot_topics_ = true;
  bool filter_joint_acceleration_ = true;
  bool filter_base_acceleration_ = true;
  bool use_imu_acceleration_ = true;
  bool use_imu_angular_velocity_ = true;
  bool imu_acceleration_includes_gravity_ = true;
  double contact_force_threshold_ = 5.0;
  double contact_timeout_ = 0.1;
  int state_sync_queue_size_ = 50;
  double state_sync_slop_ = 0.02;
  double imu_gravity_magnitude_ = 9.81;
  double joint_acceleration_process_noise_ = 0.5;
  double joint_acceleration_measurement_noise_ = 4.0;
  double joint_acceleration_initial_covariance_ = 1.0;
  double base_acceleration_process_noise_ = 0.5;
  double base_acceleration_measurement_noise_ = 4.0;
  double base_acceleration_initial_covariance_ = 1.0;

  std::vector<std::string> contact_frames_;
  std::vector<std::string> contact_topics_;
  std::vector<ContactTopicState> contact_states_;

  pinocchio::Model model_;
  std::unique_ptr<pinocchio::Data> data_;
  std::vector<pinocchio::FrameIndex> contact_frame_ids_;
  std::vector<std::string> actuated_joint_names_;
  int actuated_dof_ = 0;

  Eigen::VectorXd latest_joint_acceleration_;
  Eigen::VectorXd previous_joint_velocity_;
  std::vector<ScalarKalmanFilter> joint_acceleration_filters_;
  std::vector<ScalarKalmanFilter> base_acceleration_filters_;
  ros::Time previous_joint_stamp_;

  Eigen::VectorXd latest_base_acceleration_;
  Eigen::VectorXd previous_base_twist_;
  ros::Time previous_odom_stamp_;
  Eigen::Vector3d previous_imu_angular_velocity_ = Eigen::Vector3d::Zero();
  ros::Time previous_imu_stamp_;
};

#endif
