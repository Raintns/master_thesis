#include "a1_pinocchio_force/a1_contact_force_estimator.hpp"

#include <boost/bind.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace
{
Eigen::Quaterniond normalizeQuaternion(const geometry_msgs::Quaternion& quaternion_msg)
{
  Eigen::Quaterniond quat(
    quaternion_msg.w,
    quaternion_msg.x,
    quaternion_msg.y,
    quaternion_msg.z);

  if (quat.norm() < 1e-9) {
    return Eigen::Quaterniond::Identity();
  }

  quat.normalize();
  return quat;
}

double forceNorm(const geometry_msgs::WrenchStamped& msg)
{
  const double fx = msg.wrench.force.x;
  const double fy = msg.wrench.force.y;
  const double fz = msg.wrench.force.z;
  return std::sqrt(fx * fx + fy * fy + fz * fz);
}
}  // namespace

void A1ContactForceEstimator::ScalarKalmanFilter::configure(
  const double process,
  const double measurement,
  const double initial_covariance)
{
  process_noise = std::max(process, 1e-9);
  measurement_noise = std::max(measurement, 1e-9);
  covariance = std::max(initial_covariance, 1e-9);
  estimate = 0.0;
  initialized = false;
}

double A1ContactForceEstimator::ScalarKalmanFilter::update(const double measurement)
{
  if (!initialized) {
    estimate = measurement;
    initialized = true;
    return estimate;
  }

  covariance += process_noise;
  const double kalman_gain = covariance / (covariance + measurement_noise);
  estimate += kalman_gain * (measurement - estimate);
  covariance = (1.0 - kalman_gain) * covariance;
  return estimate;
}

void A1ContactForceEstimator::ScalarKalmanFilter::reset()
{
  estimate = 0.0;
  initialized = false;
}

A1ContactForceEstimator::A1ContactForceEstimator()
  : pnh_("~")
{
  loadParams();
  buildModel();
  initializeActuatedJointNames();

  latest_joint_acceleration_ = Eigen::VectorXd::Zero(actuated_dof_);
  previous_joint_velocity_ = Eigen::VectorXd::Zero(actuated_dof_);
  joint_acceleration_filters_.resize(static_cast<std::size_t>(actuated_dof_));
  for (auto& filter : joint_acceleration_filters_) {
    filter.configure(
      joint_acceleration_process_noise_,
      joint_acceleration_measurement_noise_,
      joint_acceleration_initial_covariance_);
  }
  latest_base_acceleration_ = Eigen::VectorXd::Zero(6);
  previous_base_twist_ = Eigen::VectorXd::Zero(6);
  base_acceleration_filters_.resize(6);
  for (auto& filter : base_acceleration_filters_) {
    filter.configure(
      base_acceleration_process_noise_,
      base_acceleration_measurement_noise_,
      base_acceleration_initial_covariance_);
  }

  contact_forces_pub_ = nh_.advertise<a1_pinocchio_force::ContactForces>(
    "a1_pinocchio_force/estimated_contact_forces", 10);
  force_magnitudes_pub_ = nh_.advertise<a1_pinocchio_force::NamedArray>(
    "a1_pinocchio_force/estimated_force_magnitudes", 10);
  active_contact_mask_pub_ = nh_.advertise<a1_pinocchio_force::NamedArray>(
    "a1_pinocchio_force/active_contact_mask", 10);

  if (publish_per_foot_topics_) {
    for (const auto& frame_name : contact_frames_) {
      per_foot_force_pubs_.push_back(
        nh_.advertise<geometry_msgs::WrenchStamped>(
          "a1_pinocchio_force/" + frame_name + "/estimated_force", 10));
    }
  }

  joint_state_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::JointState>>(
    nh_, joint_states_topic_, 100);
  odom_sub_ = std::make_unique<message_filters::Subscriber<nav_msgs::Odometry>>(
    nh_, odom_topic_, 50);
  imu_sub_ = std::make_unique<message_filters::Subscriber<sensor_msgs::Imu>>(
    nh_, imu_topic_, 100);
  state_sync_ = std::make_unique<message_filters::Synchronizer<StateSyncPolicy>>(
    StateSyncPolicy(state_sync_queue_size_), *joint_state_sub_, *odom_sub_, *imu_sub_);
  state_sync_->setMaxIntervalDuration(ros::Duration(state_sync_slop_));
  state_sync_->registerCallback(
    boost::bind(&A1ContactForceEstimator::stateCallback, this, _1, _2, _3));

  if (use_contact_topics_) {
    if (contact_topics_.size() != contact_frames_.size()) {
      ROS_WARN(
        "[a1_pinocchio_force] contact_topics and contact_frames size mismatch. "
        "Disabling contact gating.");
      use_contact_topics_ = false;
    } else {
      contact_states_.resize(contact_topics_.size());
      for (std::size_t i = 0; i < contact_topics_.size(); ++i) {
        contact_subs_.push_back(
          nh_.subscribe<geometry_msgs::WrenchStamped>(
            contact_topics_[i], 50,
            [this, i](const geometry_msgs::WrenchStamped::ConstPtr& msg) {
              contactCallback(msg, i);
            }));
      }
    }
  }

  ROS_INFO("[a1_pinocchio_force] Ready with %d actuated joints and %zu contact frames.",
           actuated_dof_, contact_frames_.size());
}

void A1ContactForceEstimator::loadParams()
{
  pnh_.param<std::string>("robot_urdf", robot_urdf_, std::string());
  pnh_.param<std::string>("joint_states_topic", joint_states_topic_, "/joint_states");
  pnh_.param<std::string>("odom_topic", odom_topic_, "/odom");
  pnh_.param<std::string>("imu_topic", imu_topic_, "/trunk_imu");
  pnh_.param<std::string>("output_frame", output_frame_, "odom");
  pnh_.param<bool>("use_contact_topics", use_contact_topics_, true);
  pnh_.param<bool>("publish_per_foot_topics", publish_per_foot_topics_, true);
  pnh_.param<bool>("filter_joint_acceleration", filter_joint_acceleration_, true);
  pnh_.param<bool>("filter_base_acceleration", filter_base_acceleration_, true);
  pnh_.param<bool>("use_imu_acceleration", use_imu_acceleration_, true);
  pnh_.param<bool>("use_imu_angular_velocity", use_imu_angular_velocity_, true);
  pnh_.param<bool>(
    "imu_acceleration_includes_gravity",
    imu_acceleration_includes_gravity_,
    true);
  pnh_.param<double>("contact_force_threshold", contact_force_threshold_, 5.0);
  pnh_.param<double>("contact_timeout", contact_timeout_, 0.1);
  pnh_.param<int>("state_sync_queue_size", state_sync_queue_size_, 50);
  pnh_.param<double>("state_sync_slop", state_sync_slop_, 0.02);
  pnh_.param<double>("imu_gravity_magnitude", imu_gravity_magnitude_, 9.81);
  pnh_.param<double>(
    "joint_acceleration_process_noise",
    joint_acceleration_process_noise_,
    0.5);
  pnh_.param<double>(
    "joint_acceleration_measurement_noise",
    joint_acceleration_measurement_noise_,
    4.0);
  pnh_.param<double>(
    "joint_acceleration_initial_covariance",
    joint_acceleration_initial_covariance_,
    1.0);
  pnh_.param<double>(
    "base_acceleration_process_noise",
    base_acceleration_process_noise_,
    0.5);
  pnh_.param<double>(
    "base_acceleration_measurement_noise",
    base_acceleration_measurement_noise_,
    4.0);
  pnh_.param<double>(
    "base_acceleration_initial_covariance",
    base_acceleration_initial_covariance_,
    1.0);

  if (!pnh_.getParam("contact_frames", contact_frames_)) {
    contact_frames_ = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
  }

  if (!pnh_.getParam("contact_topics", contact_topics_)) {
    contact_topics_ = {
      "/foot_force/FL",
      "/foot_force/FR",
      "/foot_force/RL",
      "/foot_force/RR"};
  }

  if (robot_urdf_.empty()) {
    ROS_FATAL("[a1_pinocchio_force] Parameter robot_urdf is required.");
    throw std::runtime_error("robot_urdf parameter is empty");
  }
}

void A1ContactForceEstimator::buildModel()
{
  pinocchio::urdf::buildModel(robot_urdf_, pinocchio::JointModelFreeFlyer(), model_);
  data_ = std::make_unique<pinocchio::Data>(model_);

  contact_frame_ids_.clear();
  for (const auto& frame_name : contact_frames_) {
    const pinocchio::FrameIndex frame_id = model_.getFrameId(frame_name);
    if (frame_id >= model_.frames.size()) {
      ROS_FATAL("[a1_pinocchio_force] Contact frame %s not found in URDF.", frame_name.c_str());
      throw std::runtime_error("contact frame not found");
    }
    contact_frame_ids_.push_back(frame_id);
  }
}

void A1ContactForceEstimator::initializeActuatedJointNames()
{
  actuated_joint_names_.clear();
  for (pinocchio::JointIndex joint_id = 1; joint_id < model_.njoints; ++joint_id) {
    if (model_.joints[joint_id].nq() == 1 && model_.joints[joint_id].nv() == 1) {
      actuated_joint_names_.push_back(model_.names[joint_id]);
    }
  }

  actuated_dof_ = static_cast<int>(actuated_joint_names_.size());
  if (actuated_dof_ <= 0) {
    ROS_FATAL("[a1_pinocchio_force] No actuated joints found in model.");
    throw std::runtime_error("no actuated joints");
  }
}

bool A1ContactForceEstimator::buildJointVectors(
  const sensor_msgs::JointState& msg,
  Eigen::VectorXd& joint_position,
  Eigen::VectorXd& joint_velocity,
  Eigen::VectorXd& joint_torque) const
{
  if (msg.position.empty() || msg.velocity.empty() || msg.effort.empty()) {
    return false;
  }

  std::unordered_map<std::string, std::size_t> joint_index_by_name;
  joint_index_by_name.reserve(msg.name.size());
  for (std::size_t i = 0; i < msg.name.size(); ++i) {
    joint_index_by_name[msg.name[i]] = i;
  }

  joint_position = Eigen::VectorXd::Zero(actuated_dof_);
  joint_velocity = Eigen::VectorXd::Zero(actuated_dof_);
  joint_torque = Eigen::VectorXd::Zero(actuated_dof_);

  for (int i = 0; i < actuated_dof_; ++i) {
    const auto it = joint_index_by_name.find(actuated_joint_names_[i]);
    if (it == joint_index_by_name.end()) {
      ROS_WARN_THROTTLE(
        2.0,
        "[a1_pinocchio_force] Missing joint %s in joint_states.",
        actuated_joint_names_[i].c_str());
      return false;
    }

    const std::size_t idx = it->second;
    if (idx >= msg.position.size() || idx >= msg.velocity.size() || idx >= msg.effort.size()) {
      ROS_WARN_THROTTLE(2.0, "[a1_pinocchio_force] joint_states array size mismatch.");
      return false;
    }

    joint_position[i] = msg.position[idx];
    joint_velocity[i] = msg.velocity[idx];
    joint_torque[i] = msg.effort[idx];
  }

  return true;
}

Eigen::VectorXd A1ContactForceEstimator::filterJointAcceleration(
  const Eigen::VectorXd& raw_joint_acceleration)
{
  if (!filter_joint_acceleration_) {
    return raw_joint_acceleration;
  }

  Eigen::VectorXd filtered_joint_acceleration = raw_joint_acceleration;
  const std::size_t filter_count = std::min(
    static_cast<std::size_t>(raw_joint_acceleration.size()),
    joint_acceleration_filters_.size());

  for (std::size_t i = 0; i < filter_count; ++i) {
    filtered_joint_acceleration[static_cast<Eigen::Index>(i)] =
      joint_acceleration_filters_[i].update(raw_joint_acceleration[static_cast<Eigen::Index>(i)]);
  }

  return filtered_joint_acceleration;
}

Eigen::VectorXd A1ContactForceEstimator::filterBaseAcceleration(
  const Eigen::VectorXd& raw_base_acceleration)
{
  if (!filter_base_acceleration_) {
    return raw_base_acceleration;
  }

  Eigen::VectorXd filtered_base_acceleration = raw_base_acceleration;
  const std::size_t filter_count = std::min(
    static_cast<std::size_t>(raw_base_acceleration.size()),
    base_acceleration_filters_.size());

  for (std::size_t i = 0; i < filter_count; ++i) {
    filtered_base_acceleration[static_cast<Eigen::Index>(i)] =
      base_acceleration_filters_[i].update(raw_base_acceleration[static_cast<Eigen::Index>(i)]);
  }

  return filtered_base_acceleration;
}

Eigen::Vector3d A1ContactForceEstimator::gravityVectorInBodyFrame(
  const Eigen::Quaterniond& orientation) const
{
  const Eigen::Vector3d gravity_in_world(0.0, 0.0, -imu_gravity_magnitude_);
  return orientation.conjugate() * gravity_in_world;
}

void A1ContactForceEstimator::stateCallback(
  const sensor_msgs::JointState::ConstPtr& joint_state_msg,
  const nav_msgs::Odometry::ConstPtr& odom_msg,
  const sensor_msgs::Imu::ConstPtr& imu_msg)
{
  Eigen::VectorXd joint_position;
  Eigen::VectorXd joint_velocity;
  Eigen::VectorXd joint_torque;
  if (!buildJointVectors(*joint_state_msg, joint_position, joint_velocity, joint_torque)) {
    return;
  }

  const ros::Time joint_stamp =
    joint_state_msg->header.stamp.isZero() ? ros::Time::now() : joint_state_msg->header.stamp;
  if (!previous_joint_stamp_.isZero()) {
    const double dt = (joint_stamp - previous_joint_stamp_).toSec();
    if (dt > 1e-6) {
      const Eigen::VectorXd raw_joint_acceleration =
        (joint_velocity - previous_joint_velocity_) / dt;
      latest_joint_acceleration_ = filterJointAcceleration(raw_joint_acceleration);
    }
  }

  previous_joint_stamp_ = joint_stamp;
  previous_joint_velocity_ = joint_velocity;

  const Eigen::Quaterniond odom_quat = normalizeQuaternion(odom_msg->pose.pose.orientation);
  const Eigen::Quaterniond imu_quat = normalizeQuaternion(imu_msg->orientation);

  Eigen::Vector3d base_linear_velocity(
    odom_msg->twist.twist.linear.x,
    odom_msg->twist.twist.linear.y,
    odom_msg->twist.twist.linear.z);
  Eigen::Vector3d base_angular_velocity(
    odom_msg->twist.twist.angular.x,
    odom_msg->twist.twist.angular.y,
    odom_msg->twist.twist.angular.z);
  if (use_imu_angular_velocity_) {
    base_angular_velocity << imu_msg->angular_velocity.x,
      imu_msg->angular_velocity.y,
      imu_msg->angular_velocity.z;
  }

  Eigen::VectorXd base_twist(6);
  base_twist << base_linear_velocity,
    base_angular_velocity;

  const ros::Time stamp = odom_msg->header.stamp.isZero() ? joint_stamp : odom_msg->header.stamp;
  Eigen::VectorXd raw_base_acceleration = Eigen::VectorXd::Zero(6);
  bool has_base_acceleration = false;
  if (use_imu_acceleration_) {
    raw_base_acceleration.head<3>() << imu_msg->linear_acceleration.x,
      imu_msg->linear_acceleration.y,
      imu_msg->linear_acceleration.z;
    if (imu_acceleration_includes_gravity_) {
      raw_base_acceleration.head<3>() += gravityVectorInBodyFrame(imu_quat);
    }

    const ros::Time imu_stamp = imu_msg->header.stamp.isZero() ? stamp : imu_msg->header.stamp;
    if (!previous_imu_stamp_.isZero()) {
      const double imu_dt = (imu_stamp - previous_imu_stamp_).toSec();
      if (imu_dt > 1e-6) {
        raw_base_acceleration.tail<3>() =
          (base_angular_velocity - previous_imu_angular_velocity_) / imu_dt;
        has_base_acceleration = true;
      }
    }
    previous_imu_stamp_ = imu_stamp;
    previous_imu_angular_velocity_ = base_angular_velocity;

    // Even on the first synchronized sample, the IMU linear acceleration is usable.
    has_base_acceleration = true;
  } else if (!previous_odom_stamp_.isZero()) {
    const double dt = (stamp - previous_odom_stamp_).toSec();
    if (dt > 1e-6) {
      raw_base_acceleration = (base_twist - previous_base_twist_) / dt;
      has_base_acceleration = true;
    }
  }

  if (has_base_acceleration) {
    latest_base_acceleration_ = filterBaseAcceleration(raw_base_acceleration);
  }

  previous_odom_stamp_ = stamp;
  previous_base_twist_ = base_twist;

  Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
  Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(model_.nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Zero(model_.nv);

  q.head<3>() << odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z;

  q.segment<4>(3) << odom_quat.x(), odom_quat.y(), odom_quat.z(), odom_quat.w();
  q.tail(actuated_dof_) = joint_position;

  v.head<6>() = base_twist;
  v.tail(actuated_dof_) = joint_velocity;

  a.head<6>() = latest_base_acceleration_;
  a.tail(actuated_dof_) = latest_joint_acceleration_;

  tau.tail(actuated_dof_) = joint_torque;

  pinocchio::crba(model_, *data_, q);
  Eigen::MatrixXd mass_matrix = data_->M;
  mass_matrix.triangularView<Eigen::StrictlyLower>() =
    mass_matrix.transpose().triangularView<Eigen::StrictlyLower>();

  const Eigen::VectorXd nonlinear_effects = pinocchio::nonLinearEffects(model_, *data_, q, v);

  pinocchio::forwardKinematics(model_, *data_, q, v, a);
  pinocchio::updateFramePlacements(model_, *data_);

  const std::vector<std::size_t> active_indices = activeContactIndices(stamp);
  std::vector<bool> active_mask(contact_frames_.size(), false);
  for (const auto index : active_indices) {
    active_mask[index] = true;
  }

  if (active_indices.empty()) {
    publishResults(stamp, odom_msg->header.frame_id.empty() ? output_frame_ : odom_msg->header.frame_id,
                   std::vector<Eigen::Vector3d>(contact_frames_.size(), Eigen::Vector3d::Zero()),
                   active_mask);
    return;
  }

  Eigen::MatrixXd contact_jacobian(3 * active_indices.size(), model_.nv);
  contact_jacobian.setZero();

  for (std::size_t row = 0; row < active_indices.size(); ++row) {
    pinocchio::Data::Matrix6x frame_jacobian(6, model_.nv);
    frame_jacobian.setZero();
    pinocchio::computeFrameJacobian(
      model_,
      *data_,
      q,
      contact_frame_ids_[active_indices[row]],
      pinocchio::LOCAL_WORLD_ALIGNED,
      frame_jacobian);
    contact_jacobian.block(3 * row, 0, 3, model_.nv) = frame_jacobian.topRows(3);
  }

  const Eigen::VectorXd rhs = mass_matrix * a + nonlinear_effects - tau;
  const Eigen::VectorXd solved_forces =
    contact_jacobian.transpose().completeOrthogonalDecomposition().solve(rhs);

  std::vector<Eigen::Vector3d> contact_forces(contact_frames_.size(), Eigen::Vector3d::Zero());
  for (std::size_t i = 0; i < active_indices.size(); ++i) {
    contact_forces[active_indices[i]] = solved_forces.segment<3>(3 * i);
  }

  publishResults(
    stamp,
    odom_msg->header.frame_id.empty() ? output_frame_ : odom_msg->header.frame_id,
    contact_forces,
    active_mask);
}

void A1ContactForceEstimator::contactCallback(
  const geometry_msgs::WrenchStamped::ConstPtr& msg,
  std::size_t index)
{
  if (index >= contact_states_.size()) {
    return;
  }

  contact_states_[index].received = true;
  contact_states_[index].stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
  contact_states_[index].norm = forceNorm(*msg);
  contact_states_[index].active = contact_states_[index].norm >= contact_force_threshold_;
}

std::vector<std::size_t> A1ContactForceEstimator::activeContactIndices(const ros::Time& stamp) const
{
  std::vector<std::size_t> active_indices;

  if (!use_contact_topics_) {
    active_indices.resize(contact_frames_.size());
    std::iota(active_indices.begin(), active_indices.end(), 0);
    return active_indices;
  }

  for (std::size_t i = 0; i < contact_states_.size(); ++i) {
    const auto& state = contact_states_[i];
    if (!state.received) {
      continue;
    }

    if ((stamp - state.stamp).toSec() > contact_timeout_) {
      continue;
    }

    if (state.active) {
      active_indices.push_back(i);
    }
  }

  return active_indices;
}

void A1ContactForceEstimator::publishResults(
  const ros::Time& stamp,
  const std::string& output_frame,
  const std::vector<Eigen::Vector3d>& contact_forces,
  const std::vector<bool>& contact_mask)
{
  a1_pinocchio_force::ContactForces contact_forces_msg;
  contact_forces_msg.header.stamp = stamp;
  contact_forces_msg.header.frame_id = output_frame;
  contact_forces_msg.labels = contact_frames_;

  a1_pinocchio_force::NamedArray magnitudes_msg;
  magnitudes_msg.header = contact_forces_msg.header;
  magnitudes_msg.labels = contact_frames_;

  a1_pinocchio_force::NamedArray contact_mask_msg;
  contact_mask_msg.header = contact_forces_msg.header;
  contact_mask_msg.labels = contact_frames_;

  for (std::size_t i = 0; i < contact_frames_.size(); ++i) {
    geometry_msgs::WrenchStamped force_msg;
    force_msg.header = contact_forces_msg.header;
    force_msg.wrench.force.x = contact_forces[i].x();
    force_msg.wrench.force.y = contact_forces[i].y();
    force_msg.wrench.force.z = contact_forces[i].z();
    force_msg.wrench.torque.x = 0.0;
    force_msg.wrench.torque.y = 0.0;
    force_msg.wrench.torque.z = 0.0;

    contact_forces_msg.forces.push_back(force_msg);
    magnitudes_msg.values.push_back(contact_forces[i].norm());
    contact_mask_msg.values.push_back(contact_mask[i] ? 1.0 : 0.0);

    if (publish_per_foot_topics_ && i < per_foot_force_pubs_.size()) {
      per_foot_force_pubs_[i].publish(force_msg);
    }
  }

  contact_forces_pub_.publish(contact_forces_msg);
  force_magnitudes_pub_.publish(magnitudes_msg);
  active_contact_mask_pub_.publish(contact_mask_msg);
}
