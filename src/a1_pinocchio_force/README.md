# a1_pinocchio_force

Clean A1-only Pinocchio contact-force estimation package.

## What it does

This node estimates A1 foot contact forces from:

- base pose and twist from `/odom`
- joint position, velocity, and effort from `/joint_states`
- IMU linear acceleration and angular velocity from `/trunk_imu`
- optional measured foot-force topics used only as a contact mask

It solves:

`J(q)^T * lambda = M(q) * ddq + h(q, dq) - tau`

with a least-squares solve over the active contact feet.

The estimator uses the IMU to improve the base acceleration term:

- linear acceleration comes from the IMU after gravity compensation
- angular velocity comes from the IMU
- angular acceleration is computed by differentiating IMU angular velocity

## Important note

This package estimates contact forces from robot state and torque.
It does **not** recover the controller's internal planned force command unless that command is already available in another topic.

## Topics

Published:

- `/a1_pinocchio_force/estimated_contact_forces`
- `/a1_pinocchio_force/estimated_force_magnitudes`
- `/a1_pinocchio_force/active_contact_mask`
- `/a1_pinocchio_force/<foot>/estimated_force`

## Run

```bash
source /home/rain/github_upload/devel/setup.bash
roslaunch a1_pinocchio_force a1_contact_force_estimator.launch
```

If your setup does not publish `/foot_force/*`, disable contact gating:

```bash
source /home/rain/github_upload/devel/setup.bash
roslaunch a1_pinocchio_force a1_contact_force_estimator.launch use_contact_topics:=false
```

To inspect the estimated forces:

```bash
rostopic echo /a1_pinocchio_force/estimated_contact_forces
```
