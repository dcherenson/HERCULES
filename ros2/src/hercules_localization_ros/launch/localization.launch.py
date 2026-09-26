"""Launch one cooperative-localization transport node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("agent_id", default_value="agent"),
        DeclareLaunchArgument("anchor_agent", default_value="Drone1"),
        DeclareLaunchArgument("vehicle_type", default_value=""),
        DeclareLaunchArgument(
            "algorithm",
            default_value="recursive_decentralized",
            choices=["recursive_decentralized", "gs_ci"],
        ),
        DeclareLaunchArgument("odom_local_topic", default_value=""),
        DeclareLaunchArgument("global_gps_topic", default_value=""),
        DeclareLaunchArgument("odom_origin_topic", default_value=""),
        DeclareLaunchArgument("gps_origin_topic", default_value="/hercules_drone/origin_geo_point"),
        DeclareLaunchArgument("gps_origin_topic_secondary", default_value="/hercules_ugv/origin_geo_point"),
        DeclareLaunchArgument("relative_topic", default_value=""),
        DeclareLaunchArgument("peer_topic", default_value=""),
        DeclareLaunchArgument("global_ci_topic", default_value=""),
        DeclareLaunchArgument("gps_origin_latitude", default_value="nan"),
        DeclareLaunchArgument("gps_origin_longitude", default_value="nan"),
        DeclareLaunchArgument("gps_origin_altitude", default_value="nan"),
        DeclareLaunchArgument("stale_after_sec", default_value="0.5"),
        DeclareLaunchArgument("pair_transaction_timeout_sec", default_value="0.5"),
        DeclareLaunchArgument("communication_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("camera_range_m", default_value="100.0"),
        DeclareLaunchArgument("camera_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("covariance_floor", default_value="1e-9"),
        DeclareLaunchArgument("process_covariance_floor", default_value="1e-9"),
        DeclareLaunchArgument("measurement_covariance_floor", default_value="1e-9"),
        DeclareLaunchArgument("dcl_lambda", default_value="1.0"),
        DeclareLaunchArgument("ci_self_weight", default_value="0.8"),
        DeclareLaunchArgument("unknown_motion_variance", default_value="0.25"),
        DeclareLaunchArgument("maicp_enabled", default_value="false"),
        DeclareLaunchArgument("maicp_margin", default_value="0.0"),
        DeclareLaunchArgument("maicp_covariance_gain", default_value="0.0"),
        DeclareLaunchArgument("maicp_class", default_value="unknown"),
        Node(
            package="hercules_localization_ros",
            executable="localization_node",
            name="localization_node",
            output="screen",
            parameters=[{
                "agent_id": LaunchConfiguration("agent_id"),
                "anchor_agent": LaunchConfiguration("anchor_agent"),
                "vehicle_type": LaunchConfiguration("vehicle_type"),
                "algorithm": LaunchConfiguration("algorithm"),
                "odom_local_topic": LaunchConfiguration("odom_local_topic"),
                "global_gps_topic": LaunchConfiguration("global_gps_topic"),
                "odom_origin_topic": LaunchConfiguration("odom_origin_topic"),
                "gps_origin_topic": LaunchConfiguration("gps_origin_topic"),
                "gps_origin_topic_secondary": LaunchConfiguration("gps_origin_topic_secondary"),
                "relative_topic": LaunchConfiguration("relative_topic"),
                "peer_topic": LaunchConfiguration("peer_topic"),
                "global_ci_topic": LaunchConfiguration("global_ci_topic"),
                "gps_origin_latitude": LaunchConfiguration("gps_origin_latitude"),
                "gps_origin_longitude": LaunchConfiguration("gps_origin_longitude"),
                "gps_origin_altitude": LaunchConfiguration("gps_origin_altitude"),
                "stale_after_sec": ParameterValue(LaunchConfiguration("stale_after_sec"), value_type=float),
                "pair_transaction_timeout_sec": ParameterValue(
                    LaunchConfiguration("pair_transaction_timeout_sec"), value_type=float),
                "communication_rate_hz": ParameterValue(
                    LaunchConfiguration("communication_rate_hz"), value_type=float),
                "camera_range_m": ParameterValue(LaunchConfiguration("camera_range_m"), value_type=float),
                "camera_rate_hz": ParameterValue(LaunchConfiguration("camera_rate_hz"), value_type=float),
                "covariance_floor": ParameterValue(LaunchConfiguration("covariance_floor"), value_type=float),
                "process_covariance_floor": ParameterValue(
                    LaunchConfiguration("process_covariance_floor"), value_type=float),
                "measurement_covariance_floor": ParameterValue(
                    LaunchConfiguration("measurement_covariance_floor"), value_type=float),
                "dcl_lambda": ParameterValue(LaunchConfiguration("dcl_lambda"), value_type=float),
                "ci_self_weight": ParameterValue(LaunchConfiguration("ci_self_weight"), value_type=float),
                "unknown_motion_variance": ParameterValue(
                    LaunchConfiguration("unknown_motion_variance"), value_type=float),
                "maicp_enabled": ParameterValue(
                    LaunchConfiguration("maicp_enabled"), value_type=bool),
                "maicp_margin": ParameterValue(
                    LaunchConfiguration("maicp_margin"), value_type=float),
                "maicp_covariance_gain": ParameterValue(
                    LaunchConfiguration("maicp_covariance_gain"), value_type=float),
                "maicp_class": LaunchConfiguration("maicp_class"),
            }],
        ),
    ])
