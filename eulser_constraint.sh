#!/bin/bash
cp /app/launch_dir/launch_5.py /app/ros2_ws/src/game/launch/launch.py
echo "已复制 launch_5.py 到目标目录 /app/ros2_ws/src/game/launch/"

config_name="euler_constraint"
draw_model="euler_constraint"
# 定义配置索引列表
# config_indices=("3_1" "3_2" "3_3" "3_4" "3_5" "3_6")
# config_indices=("3_11" "3_12" "3_13")
# config_indices=("3_11" "3_12" "3_13" "3_14" "3_15")
# config_indices=("c5" "c6")
config_indices=("5")
# config_indices=("3_7" "3_8" "3_9" "3_10")
# config_indices=("c1" "c2" "c3" "c4")


# 功能函数定义
start_ros_launch() {
    local index=$1
    echo "启动 ROS2 节点 (配置: $config_name@$index)..."
    cd /app/ros2_ws || exit
    colcon build
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    ros2 launch game launch.py num_nodes:=5 config:="$config_name@$index"
}

start_python_script() {
    local index=$1
    echo "启动 Python 可视化 (配置: $config_name@$index)..."
    cd /app/ros2_ws/src/game || exit
    source /opt/ros/humble/setup.bash
    source /app/ros2_ws/install/setup.bash
    python3 ./utlis/main.py "$config_name" "$index" "$draw_model"
}

# 菜单界面
echo "请选择操作模式："
echo "1) 同时启动 ROS 节点和 Python 可视化 (按顺序执行所有配置)"
echo "2) 仅启动 ROS 节点 (按顺序执行所有配置)"
echo "3) 仅启动 Python 可视化 (按顺序执行所有配置)"
read -p "输入选项 (1/2/3): " choice

case $choice in
    1)
        for index in "${config_indices[@]}"; do
            start_ros_launch "$index"
            start_python_script "$index"
        done
        ;;
    2)
        for index in "${config_indices[@]}"; do
            start_ros_launch "$index"
        done
        ;;
    3)
        for index in "${config_indices[@]}"; do
            start_python_script "$index"
        done
        ;;
    *)
        echo "无效选项！"
        exit 1
        ;;
esac

exit 0