from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('vehicle_type', default_value='drone', choices=['drone', 'ugv']),
        DeclareLaunchArgument('mode', default_value='observe', choices=['observe', 'motion', 'benchmark']),
        Node(package='hercules_control', executable='smoke_node', output='screen',
             parameters=[{'vehicle_type': LaunchConfiguration('vehicle_type'),
                          'mode': LaunchConfiguration('mode'), 'use_sim_time': False}]),
    ])
