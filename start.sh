
# config_name=exp_2
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game

# config_name=asym_4
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game

config_name=r_r
cd /app/ros2_ws
colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
cd /app/ros2_ws/src/game

# config_name=r_0
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game


# config_name=r_1
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game


# config_name=r_2
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game


# config_name=r_3
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game
# config_name=r_4
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game

# python3 ./utlis/main.py $config_name

# config_name=6
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game
# python3 ./utlis/main.py $config_name

# config_name=7
# cd /app/ros2_ws
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=fixed2@$config_name
# cd /app/ros2_ws/src/game
# python3 ./utlis/main.py $config_name
# config_name=1_1
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name

# config_name=1_2
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name


############################# paper 2
# config_name=0
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name



# config_name=0_3
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name


# config_name=0_4
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name


# config_name=0_5
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name

# config_name=2
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name

#####################

# test 2

# config_name=01
# model=euler
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$model@$config_name

##################### trigger

# config_name=0
# model=event_trigger
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=6 config:=$model@$config_name


######################## switching


# config_index=2
# config=communication
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=6 config:=$config@$config_index


######################## high_order

# config_index=0
# config=high_order
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$config@$config_index

# config_index=1
# config=high_order
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$config@$config_index

# config_index=2
# config=high_order
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$config@$config_index

# config_index=3
# config=high_order
# cd /app/ros2_ws
# rm -rf /app/ros2_ws/build
# colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# ros2 launch game launch.py num_nodes:=5 config:=$config@$config_index