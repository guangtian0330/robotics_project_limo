from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    rviz_config_dir = os.path.join(get_package_share_directory('slam_rbpf'),'rviz','slam_rbpf.rviz')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    rbpf_node = Node(
            package='slam_rbpf',
            executable='slam_rbpf_nav',
            name='slam_navigation_node',
            output='screen',
    )

    map_node = Node(
            package='slam_rbpf',
            executable='map_display',
            name='slam_map_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
    )

    path_npde = Node(
        package='slam_rbpf',
        executable='path_node',
        name='path_planner_node',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    ld = LaunchDescription()
    ld.add_action(path_npde)
    ld.add_action(rbpf_node)
    ld.add_action(map_node)

    return ld