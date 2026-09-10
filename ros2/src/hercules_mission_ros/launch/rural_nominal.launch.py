"""Complete nine-vehicle RuralAustralia mission with selectable target source."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


VEHICLES = [
    "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
    "Husky1", "Husky2", "Husky3", "Target1",
]


def generate_launch_description():
    mission_share = get_package_share_directory("hercules_mission_ros")
    wrapper_share = get_package_share_directory("airsim_ros_pkgs")
    state_config = os.path.join(mission_share, "config", "rural_nominal_state.yaml")
    wrapper_launch = os.path.join(wrapper_share, "launch", "hercules_host.launch.py")
    dry_run = LaunchConfiguration("dry_run")
    enable_target = LaunchConfiguration("enable_target")
    enable_formation = LaunchConfiguration("enable_formation")
    duration = LaunchConfiguration("duration_sec")
    log_path = LaunchConfiguration("log_path")
    target_source = LaunchConfiguration("target_source")
    observation_source = LaunchConfiguration("target_observation_source")
    distributed = IfCondition(PythonExpression([
        "'", target_source, "' == 'distributed_tracking'"
    ]))
    actions = [
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("enable_target", default_value="true"),
        DeclareLaunchArgument("enable_formation", default_value="true"),
        DeclareLaunchArgument("duration_sec", default_value="30.0"),
        DeclareLaunchArgument("target_source", default_value="distributed_tracking"),
        DeclareLaunchArgument("target_observation_source", default_value="camera"),
        DeclareLaunchArgument(
            "log_path",
            default_value="/workspaces/hercules/ros2/validation/rural_nominal/artifacts/mission.jsonl"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(wrapper_launch),
            launch_arguments={"vehicles": "both", "enable_api_control": "true",
                              "ugv_command_timeout_sec": "0.5"}.items()),
        Node(package="hercules_mission_ros", executable="direct_pose_bridge",
             name="mission_direct_pose_bridge", output="screen",
             parameters=[{"vehicle_names": VEHICLES}]),
        Node(package="hercules_mission_ros", executable="state_node",
             name="mission_state_adapter", output="screen", parameters=[state_config]),
        Node(package="hercules_mission_ros", executable="rural_nominal_mission_node",
             name="rural_nominal_mission", output="screen",
             parameters=[{"dry_run": ParameterValue(dry_run, value_type=bool),
                          "enable_target": ParameterValue(enable_target, value_type=bool),
                          "enable_formation": ParameterValue(enable_formation, value_type=bool),
                          "duration_sec": ParameterValue(duration, value_type=float),
                          "target_source": target_source,
                          "target_observation_source": observation_source,
                          "route_heading_rad": -1.5083775167989393,
                          "log_path": log_path}]),
    ]
    actions.append(Node(
        package="hercules_tracking_ros", executable="target_observer_node",
        name="target_observer", output="screen", condition=distributed,
        parameters=[{"observation_source": observation_source,
                     "tracking_rate": 4.0,
                     "tracking_measurement_std": 0.25,
                     "target_sensing_range": 100.0,
                     "truth_seed": 7}],
    ))
    for agent in VEHICLES[:-1]:
        actions.append(Node(
            package="hercules_tracking_ros", executable="tracker_node",
            namespace=f"hercules_tracking/{agent}", name="target_tracker",
            output="screen", condition=distributed,
            parameters=[{"agent_id": agent, "target_id": "Target1",
                         "tracking_window": 5.0,
                         "tracking_admm_rho": 1.0,
                         "tracking_admm_max_iterations": 20,
                         "tracking_admm_tolerance": 0.001,
                         "tracking_process_noise": 0.20,
                         "tracking_measurement_std": 0.25,
                         "round_timeout_sec": 0.008,
                         "seed_timeout_sec": 0.020,
                         "measurement_wait_sec": 0.010}],
        ))
    return LaunchDescription(actions)
