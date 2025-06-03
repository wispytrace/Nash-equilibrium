#!/bin/bash

config_index=0
config_name="euler_mq"
draw_model="euler_constraint"

# 功能函数定义
start_ros_launch() {
    echo "启动 ROS2 节点..."
    cd /app/ros2_ws || exit
    colcon build
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    ros2 launch game launch.py num_nodes:=5 config:="$config_name@$config_index"
}

start_python_script() {
    echo "启动 Python 可视化..."
    cd /app/ros2_ws/src/game || exit
    source /opt/ros/humble/setup.bash
    source /app/ros2_ws/install/setup.bash
    python3 ./utlis/main.py "$config_name" "$config_index" "$draw_model"
}

# 菜单界面
echo "请选择操作模式："
echo "1) 同时启动 ROS 节点和 Python 可视化"
echo "2) 仅启动 ROS 节点"
echo "3) 仅启动 Python 可视化"
read -p "输入选项 (1/2/3): " choice

case $choice in
    1)
        start_ros_launch     # 后台运行ROS节点
        start_python_script
        ;;
    2)
        start_ros_launch
        ;;
    3)
        start_python_script
        ;;
    *)
        echo "无效选项！"
        exit 1
        ;;
esac

exit 0