colcon clean
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch game launch.py num_nodes:=3 config:=test@0