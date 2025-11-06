from launch import LaunchDescription
from launch_ros.actions import Node
from launch import LaunchContext

from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
import sys

node_nums = 6

def generate_launch_description():
    
    config = DeclareLaunchArgument(
        'config',
        default_value='fixed@0',
        description='game model conig')
    
    config = LaunchConfiguration('config')
    

    nodes = []
    for i in range(node_nums):
        node = Node(
            package='game',
            executable='node',
            name=f'node_{i}',
            output='screen',
            parameters=[
                {'node_id': i},
                {'config': config}
            ],
        )
        nodes.append(node)
    
    nodes.append(Node(
        package='game',
        executable='sync',
        name=f'sync',
        output='screen',
        parameters=[
            {'config': config},
            {'node_nums': node_nums}
        ],
    ))

    return LaunchDescription(nodes)

