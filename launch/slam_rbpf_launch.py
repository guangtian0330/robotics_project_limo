from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_config_file = os.path.join(
        get_package_share_directory('slam_rbpf'),
        'config',
        'slam_rbpf.rviz'
    )
    return LaunchDescription([
        Node(
            package='slam_rbpf',
            executable='slam_rbpf_nav',
            name='slam_navigation_node',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        ),
    ])
