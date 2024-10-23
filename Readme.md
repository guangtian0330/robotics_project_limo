## 1. Package Introduction
### Please follow the procedure to sync the latest code of the project.
1. git clone git@github.com:guangtian0330/robotics_project_limo.git
2. git checkout master
3. git pull
### The file structure is as follows:
```plaintext
├── launch/
│   ├── gazebo_models_diff.launch.py     # Launch file for Gazebo models
│   ├── slam_rbpf_gazebo.launch.py       # Launch file for SLAM RBPF in Gazebo
│   └── slam_rbpf_limo.launch.py         # Launch file for SLAM RBPF on LIMO robot
├── log/                                 # Log files directory
├── meshes/                              # 3D meshes for robot or environment models
├── resource/                            # Additional resource files
├── rviz/                                # RViz configuration files
├── slam_rbpf/
│   ├── __init__.py                      # Initialization script for the package
│   ├── map_display.py                   # Map display utilities
│   ├── path_planner_bak.py              # Backup path planner
│   ├── path_planner.py                  # Path planning implementation
│   ├── rbpf_particle.py                 # RBPF particle filter implementation
│   ├── rbpf_SLAM.py                     # SLAM implementation using RBPF
│   ├── scan_matching.py                 # Scan matching for localization
│   ├── slam_nav.py                      # SLAM navigation control
│   ├── transformations.py               # Utility for coordinate transformations
│   ├── update_models.py                 # Model update script
│   ├── utils.py                         # General utility functions
│   └── vel_cmd_node.py                  # Node for velocity command control, used in limo robotics
├── urdf/                                # URDF files for robot description
├── worlds/
│   ├── basic.world                      # Basic Gazebo world for testing
│   ├── simple_hallway.world             # Simple hallway world for navigation
│   ├── simple_hallway2.world            # Alternate hallway world
│   ├── simple_world.world               # Simple Gazebo world for simulation
│   └── warehouse.world                  # deprecated.
├── package.xml                          # ROS2 package manifest
├── Readme.md                            # Project README file
├── setup.cfg                            # Configuration for packaging
└── setup.py                             # Setup script for packaging
```
## 2. Building and Launching procedure.
### Launching package in simulation:
1. Build the rbpf package:
```
# This is used to colcon build the slam_rbpf package after modifying. 
# If you want to build limo_description after changing the map, you need to do: sudo colcon build --packages-select limo_description
sudo colcon build --packages-select slam_rbpf
```
2. launch the gazebo map using limo_description:
```
ros2 launch limo_description gazebo_models_diff.launch.py
```
3. After the gazebo is displayed, launch rbpf package using: 
```
ros2 launch slam_rbpf slam_rbpf_limo.launch.py
```
### Launching package in limo:
1. Launch the lidar, odometer and other necessary packages using:
```
ros2 launch limo_bringup limo_start.launch.py
```
2. Show the topic list and check if there'e everything you need:
```
ros2 topic list
```
3. Make sure slam_nav.py is using '/cmd_vel_input' instead of '/cmd_vel/ in the code, and it's using odom_subscription with '/wheel/odom' instead of '/odometry' for gazebo simulation
4. Make sure VelocityPublisher in vel_cmd_node.py has started the timer for velocity publishing.
5. Launch the slam package:
```
ros2 launch slam_rbpf slam_rbpf_limo.launch.py
```