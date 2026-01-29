# master_thesis (AlienGo + Wild Visual Navigation)

## Quick Start

### 0) Prerequirement
Install Pinocchio
```bash
sudo apt update
sudo apt install -y robotpkg-pinocchio robotpkg-py38-pinocchio
```

### 1) Clone 
```bash
git clone https://github.com/Raintns/master_thesis.git
cd master_thesis
```

### 2) install dependencies
```bash
source /opt/ros/noetic/setup.bash
sudo apt install -y python3-rosdep
rosdep install --from-paths src --ignore-src -r -y
cd assets
pip3 install -e ./self_supervised_segmentation
cd ~/master_thesis/src
pip3 install -e ./wild_visual_navigation
```

May have following issue:
1)failing building wandb because of missing Go binary
```bash
install Go binary first:
sudo apt install -y golang-go
```
2)missing building wheel for wandb
```bash
python -m pip install -U pip setuptools wheel
```
3)if liegroups install fail:
```bash
python -m pip install "liegroups @ git+https://github.com/mmattamala/liegroups "
```

### 3) build the ros workspace
```bash
source source /opt/ros/noetic/setup.bash
catkin build
```

### 4) running code:
in separate terminal run the following code:
```bash
cd master_thesis
source devel/setup.bash
roslaunch aliengo_wild_visual_navigation aliengo_sim.launch 2> >(grep -v TF_REPEATED_DATA buffer_core) 

roslaunch aliengo_dynamics_computer combined_force_publisher.launch 2> >(grep -v TF_REPEATED_DATA buffer_core)

roslaunch aliengo_wild_visual_navigation aliengo_wild_visual_navigation.launch 2> >(grep -v TF_REPEATED_DATA buffer_core)

roslaunch champ_teleop teleop.launch
```
for the first time running with 2> >(grep -v TF_REPEATED_DATA buffer_core) will have error,so for the first time running just run without this half.

### 5) further more possible issue:
5.1 the gazebo world may missing the grasspatch model which u can find in the assets folder:
if missing just copy paste this folder in to your local gazebo path, normally just under .gazebo/models
5.2 if there have missing the rviz-grid-map:
```bash
sudo apt install ros-noetic-grid-map-rviz-plugin
```

### 6)reference
https://github.com/leggedrobotics/wild_visual_navigation
https://github.com/chvmp/champ.git

