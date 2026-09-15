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
    cbf_share = get_package_share_directory("hercules_cbf_ros")
    wrapper_share = get_package_share_directory("airsim_ros_pkgs")
    state_config = os.path.join(mission_share, "config", "rural_nominal_state.yaml")
    default_mission_config = os.path.join(mission_share, "config", "rural_tracking_no_cbf.yaml")
    wrapper_launch = os.path.join(wrapper_share, "launch", "hercules_host.launch.py")
    dry_run = LaunchConfiguration("dry_run")
    enable_target = LaunchConfiguration("enable_target")
    enable_formation = LaunchConfiguration("enable_formation")
    duration = LaunchConfiguration("duration_sec")
    log_path = LaunchConfiguration("log_path")
    target_source = LaunchConfiguration("target_source")
    observation_source = LaunchConfiguration("target_observation_source")
    cbf_enabled = LaunchConfiguration("cbf_enabled")
    cbf_method = LaunchConfiguration("cbf_method")
    cbf_obstacle_source = LaunchConfiguration("cbf_obstacle_source")
    uncertainty_radius = LaunchConfiguration("uncertainty_radius")
    actuation_profile = LaunchConfiguration("actuation_profile")
    mission_config = LaunchConfiguration("mission_config")
    cbf_config = LaunchConfiguration("cbf_config")
    enable_collision_observer = LaunchConfiguration("enable_collision_observer")
    record_video = LaunchConfiguration("record_video")
    video_output_dir = LaunchConfiguration("video_output_dir")
    video_staging_dir = LaunchConfiguration("video_staging_dir")
    record_uav = LaunchConfiguration("record_uav")
    record_ugv = LaunchConfiguration("record_ugv")
    video_fps = LaunchConfiguration("video_fps")
    video_width = LaunchConfiguration("video_width")
    video_height = LaunchConfiguration("video_height")
    gif_fps = LaunchConfiguration("gif_fps")
    gif_height = LaunchConfiguration("gif_height")
    playback_speed = LaunchConfiguration("playback_speed")
    video_keep_frames = LaunchConfiguration("video_keep_frames")
    route_heading = LaunchConfiguration("route_heading_rad")
    target_speed = LaunchConfiguration("target_speed")
    target_pattern_length = LaunchConfiguration("target_pattern_length")
    target_pattern_width = LaunchConfiguration("target_pattern_width")
    target_sample_count = LaunchConfiguration("target_sample_count")
    target_start_sample_index = LaunchConfiguration("target_start_sample_index")
    target_direction = LaunchConfiguration("target_direction")
    target_waypoint_radius = LaunchConfiguration("target_waypoint_radius")
    target_heading_gain = LaunchConfiguration("target_heading_gain")
    target_max_yaw_rate = LaunchConfiguration("target_max_yaw_rate")
    target_minimum_alignment = LaunchConfiguration("target_minimum_alignment")
    target_center_x = LaunchConfiguration("target_center_x")
    target_center_y = LaunchConfiguration("target_center_y")
    target_center_z = LaunchConfiguration("target_center_z")
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
        DeclareLaunchArgument("cbf_enabled", default_value="false"),
        DeclareLaunchArgument("cbf_method", default_value="mestres"),
        DeclareLaunchArgument("cbf_obstacle_source", default_value="none"),
        DeclareLaunchArgument("uncertainty_radius", default_value="0.0"),
        DeclareLaunchArgument("actuation_profile", default_value="current_ros"),
        DeclareLaunchArgument("mission_config", default_value=default_mission_config),
        DeclareLaunchArgument("cbf_config", default_value=os.path.join(
            mission_share, "config", "rural_cbf.yaml")),
        DeclareLaunchArgument("enable_collision_observer", default_value="false"),
        DeclareLaunchArgument("record_video", default_value="false"),
        DeclareLaunchArgument(
            "video_output_dir",
            default_value="/workspaces/hercules/ros2/validation/rural_nominal/artifacts/video"),
        DeclareLaunchArgument("video_staging_dir", default_value=""),
        DeclareLaunchArgument("record_uav", default_value="Drone1"),
        DeclareLaunchArgument("record_ugv", default_value="Husky1"),
        DeclareLaunchArgument("video_fps", default_value="30.0"),
        DeclareLaunchArgument("video_width", default_value="1280"),
        DeclareLaunchArgument("video_height", default_value="720"),
        DeclareLaunchArgument("gif_fps", default_value="10.0"),
        DeclareLaunchArgument("gif_height", default_value="540"),
        DeclareLaunchArgument("playback_speed", default_value="2.0"),
        DeclareLaunchArgument("video_keep_frames", default_value="false"),
        DeclareLaunchArgument("route_heading_rad", default_value="-1.5083775167989393"),
        DeclareLaunchArgument("target_speed", default_value="0.10"),
        DeclareLaunchArgument("target_pattern_length", default_value="10.0"),
        DeclareLaunchArgument("target_pattern_width", default_value="8.0"),
        DeclareLaunchArgument("target_sample_count", default_value="64"),
        DeclareLaunchArgument("target_start_sample_index", default_value="5"),
        DeclareLaunchArgument("target_direction", default_value="1"),
        DeclareLaunchArgument("target_waypoint_radius", default_value="1.0"),
        DeclareLaunchArgument("target_heading_gain", default_value="2.0"),
        DeclareLaunchArgument("target_max_yaw_rate", default_value="1.5"),
        DeclareLaunchArgument("target_minimum_alignment", default_value="0.75"),
        DeclareLaunchArgument("target_center_x", default_value="nan"),
        DeclareLaunchArgument("target_center_y", default_value="nan"),
        DeclareLaunchArgument("target_center_z", default_value="nan"),
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
             parameters=[cbf_config, mission_config, {"dry_run": ParameterValue(dry_run, value_type=bool),
                          "enable_target": ParameterValue(enable_target, value_type=bool),
                          "enable_formation": ParameterValue(enable_formation, value_type=bool),
                          "duration_sec": ParameterValue(duration, value_type=float),
                          "target_source": target_source,
                          "target_observation_source": observation_source,
                          "cbf_enabled": ParameterValue(cbf_enabled, value_type=bool),
                          "cbf_method": cbf_method,
                          "cbf_obstacle_source": cbf_obstacle_source,
                          "uncertainty_radius": ParameterValue(uncertainty_radius, value_type=float),
                          "actuation_profile": actuation_profile,
                          "route_heading_rad": ParameterValue(route_heading, value_type=float),
                          "target_speed": ParameterValue(target_speed, value_type=float),
                          "target_pattern_length": ParameterValue(target_pattern_length, value_type=float),
                          "target_pattern_width": ParameterValue(target_pattern_width, value_type=float),
                          "target_sample_count": ParameterValue(target_sample_count, value_type=int),
                          "target_start_sample_index": ParameterValue(target_start_sample_index, value_type=int),
                          "target_direction": ParameterValue(target_direction, value_type=int),
                          "target_waypoint_radius": ParameterValue(target_waypoint_radius, value_type=float),
                          "target_heading_gain": ParameterValue(target_heading_gain, value_type=float),
                          "target_max_yaw_rate": ParameterValue(target_max_yaw_rate, value_type=float),
                          "target_minimum_alignment": ParameterValue(target_minimum_alignment, value_type=float),
                          "target_center_x": ParameterValue(target_center_x, value_type=float),
                          "target_center_y": ParameterValue(target_center_y, value_type=float),
                          "target_center_z": ParameterValue(target_center_z, value_type=float),
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
    actions.append(Node(
        package="hercules_cbf_ros", executable="obstacle_observer_node",
        name="obstacle_observer", output="screen",
        condition=IfCondition(PythonExpression([
            "'", cbf_obstacle_source, "' == 'perception'"
        ])),
        parameters=[os.path.join(cbf_share, "config", "rural_perception.yaml")],
    ))
    actions.append(Node(
        package="hercules_cbf_ros", executable="mission_collision_observer_node",
        name="mission_collision_observer", output="screen",
        condition=IfCondition(enable_collision_observer),
        parameters=[os.path.join(cbf_share, "config", "rural_perception.yaml")],
    ))
    actions.append(Node(
        package="hercules_mission_ros", executable="ros_video_recorder",
        name="ros_video_recorder", output="screen",
        condition=IfCondition(record_video),
        parameters=[{
            "log_path": log_path,
            "video_output_dir": video_output_dir,
            "video_staging_dir": video_staging_dir,
            "record_uav": record_uav,
            "record_ugv": record_ugv,
            "video_fps": ParameterValue(video_fps, value_type=float),
            "video_width": ParameterValue(video_width, value_type=int),
            "video_height": ParameterValue(video_height, value_type=int),
            "gif_fps": ParameterValue(gif_fps, value_type=float),
            "gif_height": ParameterValue(gif_height, value_type=int),
            "playback_speed": ParameterValue(playback_speed, value_type=float),
            "duration_sec": ParameterValue(duration, value_type=float),
            "video_keep_frames": ParameterValue(video_keep_frames, value_type=bool),
            "route_heading_rad": ParameterValue(route_heading, value_type=float),
        }],
    ))
    return LaunchDescription(actions)
