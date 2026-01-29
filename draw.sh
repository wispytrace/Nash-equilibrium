# config_index=2
# config_name="euler"
# draw_model="euler"
# cd ros2_ws
# # colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# cd /app/ros2_ws/src/game
# python3 ./utlis/main.py $config_name $config_index $draw_model


# switching
# config_index=2
# config_name="communication"
# draw_model="switching"
# cd ros2_ws
# # colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# cd /app/ros2_ws/src/game
# python3 ./utlis/main.py $config_name $config_index $draw_model


# high-order
# config_index=0
# config_name="communication"
# draw_model="switching"
# cd ros2_ws
# # colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# cd /app/ros2_ws/src/game
# python3 ./utlis/main.py $config_name $config_index $draw_model

# config_index=2
# config_name="high_order"
# draw_model="high_order"
# cd ros2_ws
# # colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# cd /app/ros2_ws/src/game
# python3 ./utlis/main.py $config_name $config_index $draw_model

# config_index=r_0
# config_name="fixed2"
# draw_model="fixed"
# cd ros2_ws
# # colcon build
# source /opt/ros/humble/setup.bash
# source install/setup.bash
# cd /app/ros2_ws/src/game
# python3 ./utlis/main.py $config_name $config_index $draw_model

config_index=r_1
config_name="fixed2"
draw_model="fixed"
cd ros2_ws
# colcon build
source /opt/ros/humble/setup.bash
source install/setup.bash
cd /app/ros2_ws/src/game
python3 ./utlis/main.py $config_name $config_index $draw_model
