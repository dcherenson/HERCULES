from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("hercules_mission_ros"),
        "config", "hero_smoke_state.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        Node(
            package="hercules_mission_ros",
            executable="state_node",
            name="mission_state_adapter",
            output="screen",
            parameters=[LaunchConfiguration("config")],
        ),
    ])
