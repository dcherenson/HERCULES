"""Launch the six independent RuralAustralia Target1 trackers."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


UAV_AGENTS = ["Drone1", "Drone2", "SimpleFlight"]
UGV_AGENTS = ["Husky1", "Husky2", "Husky3"]
AGENTS = UAV_AGENTS + UGV_AGENTS


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("maicp_enabled", default_value="false"),
        DeclareLaunchArgument("maicp_margin", default_value="0.0"),
        DeclareLaunchArgument("maicp_covariance_gain", default_value="0.0"),
        *[Node(
            package="hercules_tracking_ros",
            executable="tracker_node",
            namespace=f"hercules_tracking/{agent}",
            name="target_tracker",
            output="screen",
            parameters=[{
                "agent_id": agent,
                "maicp_enabled": ParameterValue(
                    LaunchConfiguration("maicp_enabled"), value_type=bool),
                "maicp_margin": ParameterValue(
                    LaunchConfiguration("maicp_margin"), value_type=float),
                "maicp_covariance_gain": ParameterValue(
                    LaunchConfiguration("maicp_covariance_gain"), value_type=float),
                "maicp_class": "uav" if agent in UAV_AGENTS else "ugv",
            }],
        ) for agent in AGENTS]
    ])
