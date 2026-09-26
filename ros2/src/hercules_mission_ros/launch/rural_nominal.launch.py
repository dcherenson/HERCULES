"""Complete seven-vehicle RuralAustralia mission with selectable target source."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


UAVS = ["Drone1", "Drone2", "SimpleFlight"]
VEHICLES = [
    "Drone1", "Drone2", "SimpleFlight",
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
    truth_obstacle_fixture = LaunchConfiguration("truth_obstacle_fixture")
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
    video_rpc_port = LaunchConfiguration("video_rpc_port")
    video_car_port = LaunchConfiguration("video_car_port")
    airsim_host = LaunchConfiguration("airsim_host")
    drone_port = LaunchConfiguration("drone_port")
    ugv_port = LaunchConfiguration("ugv_port")
    is_vulkan = LaunchConfiguration("is_vulkan")
    publish_clock = LaunchConfiguration("publish_clock")
    use_sim_time = LaunchConfiguration("use_sim_time")
    state_freshness_timeout = LaunchConfiguration("state_freshness_timeout_sec")
    startup_timeout = LaunchConfiguration("startup_timeout_sec")
    truth_rng_mode = LaunchConfiguration("truth_rng_mode")
    localization_algorithm = LaunchConfiguration("localization_algorithm")
    localization_anchor_agent = LaunchConfiguration("localization_anchor_agent")
    localization_source = LaunchConfiguration("localization_source")
    control_source = LaunchConfiguration("control_source")
    localization_stale_after = LaunchConfiguration("localization_stale_after_sec")
    localization_transaction_timeout = LaunchConfiguration("localization_transaction_timeout_sec")
    localization_covariance_floor = LaunchConfiguration("localization_covariance_floor")
    localization_process_covariance_floor = LaunchConfiguration("localization_process_covariance_floor")
    localization_measurement_covariance_floor = LaunchConfiguration(
        "localization_measurement_covariance_floor")
    dcl_lambda = LaunchConfiguration("dcl_lambda")
    ci_self_weight = LaunchConfiguration("ci_self_weight")
    unknown_motion_variance = LaunchConfiguration("unknown_motion_variance")
    localization_relative_rate = LaunchConfiguration("localization_relative_rate_hz")
    localization_communication_rate = LaunchConfiguration("localization_communication_rate_hz")
    localization_camera_range = LaunchConfiguration("localization_camera_range_m")
    localization_observation_source = LaunchConfiguration("localization_observation_source")
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
    maicp_case = LaunchConfiguration("maicp_case")
    maicp_steps = LaunchConfiguration("maicp_steps")
    maicp_margin_ugv = LaunchConfiguration("maicp_margin_ugv")
    maicp_margin_uav = LaunchConfiguration("maicp_margin_uav")
    maicp_model_ugv = LaunchConfiguration("maicp_model_ugv")
    maicp_model_uav = LaunchConfiguration("maicp_model_uav")
    maicp_gain_ugv = LaunchConfiguration("maicp_gain_ugv")
    maicp_gain_uav = LaunchConfiguration("maicp_gain_uav")
    tracking_iterations = LaunchConfiguration("tracking_iterations")
    tracking_fixed_iterations = LaunchConfiguration("tracking_fixed_iterations")
    truth_seed = LaunchConfiguration("truth_seed")
    actions = [
        DeclareLaunchArgument("maicp_case", default_value=""),
        DeclareLaunchArgument("maicp_steps", default_value="100"),
        DeclareLaunchArgument("maicp_margin_ugv", default_value="0.0"),
        DeclareLaunchArgument("maicp_margin_uav", default_value="0.0"),
        DeclareLaunchArgument("maicp_model_ugv", default_value="0 0 0 0 0 0"),
        DeclareLaunchArgument("maicp_model_uav", default_value="0 0 0 0 0 0"),
        DeclareLaunchArgument("maicp_gain_ugv", default_value="0.30"),
        DeclareLaunchArgument("maicp_gain_uav", default_value="0.20"),
        DeclareLaunchArgument("tracking_iterations", default_value="20"),
        DeclareLaunchArgument("tracking_fixed_iterations", default_value="false"),
        DeclareLaunchArgument("truth_seed", default_value="7"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("enable_target", default_value="true"),
        DeclareLaunchArgument("enable_formation", default_value="true"),
        DeclareLaunchArgument("duration_sec", default_value="30.0"),
        DeclareLaunchArgument("target_source", default_value="distributed_tracking"),
        DeclareLaunchArgument("target_observation_source", default_value="camera"),
        DeclareLaunchArgument("cbf_enabled", default_value="false"),
        DeclareLaunchArgument("cbf_method", default_value="mestres"),
        DeclareLaunchArgument("cbf_obstacle_source", default_value="none"),
        DeclareLaunchArgument("truth_obstacle_fixture", default_value="false"),
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
        DeclareLaunchArgument("video_rpc_port", default_value="41451"),
        DeclareLaunchArgument("video_car_port", default_value="41452"),
        DeclareLaunchArgument("airsim_host", default_value=os.environ.get("AIRSIM_HOST", "127.0.0.1")),
        DeclareLaunchArgument("drone_port", default_value=os.environ.get("AIRSIM_MULTIROTOR_PORT", "41451")),
        DeclareLaunchArgument("ugv_port", default_value=os.environ.get("AIRSIM_CAR_PORT", "41452")),
        DeclareLaunchArgument("is_vulkan", default_value="true"),
        DeclareLaunchArgument("publish_clock", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("state_freshness_timeout_sec", default_value="0.5"),
        # Startup can be slow on a host-shared Unreal/Docker session while
        # seven AirSim wrappers connect and the state adapter receives two
        # advancing samples per vehicle.  Keep the node's normal 30 s default
        # unless the caller explicitly requests a longer trial grace period.
        DeclareLaunchArgument("startup_timeout_sec", default_value="30.0"),
        DeclareLaunchArgument("truth_rng_mode", default_value="seeded",
                              choices=["seeded", "random", "python_global"]),
        DeclareLaunchArgument("localization_algorithm", default_value="recursive_decentralized",
                              choices=["recursive_decentralized", "gs_ci"]),
        DeclareLaunchArgument("localization_anchor_agent", default_value="Drone1"),
        DeclareLaunchArgument("localization_source", default_value="estimate",
                              choices=["estimate", "truth"]),
        DeclareLaunchArgument("control_source", default_value=""),
        DeclareLaunchArgument("localization_stale_after_sec", default_value="0.5"),
        DeclareLaunchArgument("localization_transaction_timeout_sec", default_value="0.5"),
        DeclareLaunchArgument("localization_covariance_floor", default_value="1e-9"),
        DeclareLaunchArgument("localization_process_covariance_floor", default_value="1e-9"),
        DeclareLaunchArgument("localization_measurement_covariance_floor", default_value="1e-9"),
        DeclareLaunchArgument("dcl_lambda", default_value="1.0"),
        DeclareLaunchArgument("ci_self_weight", default_value="0.8"),
        DeclareLaunchArgument("unknown_motion_variance", default_value="0.25"),
        DeclareLaunchArgument("localization_relative_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("localization_communication_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("localization_camera_range_m", default_value="100.0"),
        DeclareLaunchArgument("localization_observation_source", default_value="camera",
                             choices=["camera", "truth"]),
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
            launch_arguments={"vehicles": "both", "host_ip": airsim_host,
                              "drone_port": drone_port, "ugv_port": ugv_port,
                              "is_vulkan": is_vulkan,
                              "publish_clock": publish_clock,
                              "use_sim_time": use_sim_time,
                              "enable_api_control": "true",
                              "ugv_command_timeout_sec": "0.5"}.items()),
        Node(package="hercules_mission_ros", executable="direct_pose_bridge",
             name="mission_direct_pose_bridge", output="screen",
             parameters=[{"vehicle_names": VEHICLES, "rpc_host": airsim_host,
                          "rpc_port": ParameterValue(drone_port, value_type=int),
                          "car_port": ParameterValue(ugv_port, value_type=int)}]),
        Node(package="hercules_mission_ros", executable="state_node",
             name="mission_state_adapter", output="screen",
             parameters=[state_config, {
                 "freshness_timeout_sec": ParameterValue(
                     state_freshness_timeout, value_type=float)
             }]),
        Node(package="hercules_mission_ros", executable="rural_nominal_mission_node",
             name="rural_nominal_mission", output="screen",
             parameters=[cbf_config, mission_config, {"dry_run": ParameterValue(dry_run, value_type=bool),
                          "enable_target": ParameterValue(enable_target, value_type=bool),
                          "maicp_case": maicp_case,
                          "maicp_steps": ParameterValue(maicp_steps, value_type=int),
                          "cbf_uav_velocity_limit": ParameterValue(PythonExpression([
                              "5.0 if '", maicp_case, "' else 3.0"]), value_type=float),
                          "cbf_ugv_speed_limit": ParameterValue(PythonExpression([
                              "5.0 if '", maicp_case, "' else 3.0"]), value_type=float),
                          "cbf_ugv_acceleration_limit": ParameterValue(PythonExpression([
                              "6.0 if '", maicp_case, "' else 3.0"]), value_type=float),
                          "maicp_margin_ugv": ParameterValue(maicp_margin_ugv, value_type=float),
                          "maicp_margin_uav": ParameterValue(maicp_margin_uav, value_type=float),
                          "maicp_model_ugv": ParameterValue(maicp_model_ugv, value_type=str),
                          "maicp_model_uav": ParameterValue(maicp_model_uav, value_type=str),
                          "enable_formation": ParameterValue(enable_formation, value_type=bool),
                          "duration_sec": ParameterValue(duration, value_type=float),
                          "target_source": target_source,
                          "target_observation_source": observation_source,
                          "cbf_enabled": ParameterValue(cbf_enabled, value_type=bool),
                          "cbf_method": cbf_method,
                          "cbf_obstacle_source": cbf_obstacle_source,
                          "truth_obstacle_fixture": ParameterValue(truth_obstacle_fixture, value_type=bool),
                          "uncertainty_radius": ParameterValue(uncertainty_radius, value_type=float),
                          "localization_algorithm": localization_algorithm,
                          "localization_source": localization_source,
                          "control_source": control_source,
                          "localization_stale_after_sec": ParameterValue(localization_stale_after, value_type=float),
                          "actuation_profile": actuation_profile,
                          "route_heading_rad": ParameterValue(route_heading, value_type=float),
                          "target_speed": ParameterValue(target_speed, value_type=float),
                          "target_pattern_length": ParameterValue(target_pattern_length, value_type=float),
                          "target_pattern_width": ParameterValue(target_pattern_width, value_type=float),
                          "target_sample_count": ParameterValue(target_sample_count, value_type=int),
                          "startup_timeout_sec": ParameterValue(startup_timeout, value_type=float),
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
    for agent in VEHICLES[:-1]:
        vehicle_prefix = "hercules_drone" if agent in UAVS else "hercules_ugv"
        actions.append(Node(
            package="hercules_localization_ros", executable="localization_node",
            name=f"localization_{agent}", output="screen",
            condition=IfCondition(PythonExpression(["'", localization_source, "' == 'estimate'"])),
            parameters=[{
                "agent_id": agent,
                "maicp_enabled": ParameterValue(PythonExpression([
                    "'", maicp_case, "' == 'localization'"]), value_type=bool),
                "maicp_class": "uav" if agent in UAVS else "ugv",
                "maicp_margin": ParameterValue(PythonExpression([
                    "'", maicp_case, "' == 'localization' and ",
                    maicp_margin_uav if agent in UAVS else maicp_margin_ugv, " or 0.0"]), value_type=float),
                "maicp_covariance_gain": ParameterValue(
                    maicp_gain_uav if agent in UAVS else maicp_gain_ugv, value_type=float),
                "anchor_agent": localization_anchor_agent,
                "algorithm": localization_algorithm,
                "odom_local_topic": f"/{vehicle_prefix}/{agent}/ground_truth/odom_local",
                "global_gps_topic": f"/{vehicle_prefix}/{agent}/global_gps",
                "odom_origin_topic": f"/hercules_mission/calibrated_origin/{agent}",
                "gps_origin_topic": "/hercules_drone/origin_geo_point",
                "gps_origin_topic_secondary": "/hercules_ugv/origin_geo_point",
                "stale_after_sec": ParameterValue(localization_stale_after, value_type=float),
                "pair_transaction_timeout_sec": ParameterValue(
                    localization_transaction_timeout, value_type=float),
                "covariance_floor": ParameterValue(
                    localization_covariance_floor, value_type=float),
                "process_covariance_floor": ParameterValue(
                    localization_process_covariance_floor, value_type=float),
                "measurement_covariance_floor": ParameterValue(
                    localization_measurement_covariance_floor, value_type=float),
                "camera_range_m": ParameterValue(localization_camera_range, value_type=float),
                "camera_rate_hz": ParameterValue(localization_relative_rate, value_type=float),
                "relative_topic": "/hercules_localization/relative",
                "peer_topic": "/hercules_localization/peer_estimate",
                "global_ci_topic": "/hercules_localization/gs_ci",
                "peer_output_topic": "/hercules_localization/peer_estimate",
                "global_agent_ids": VEHICLES[:-1],
                "communication_rate_hz": ParameterValue(
                    localization_communication_rate, value_type=float),
                "dcl_lambda": ParameterValue(dcl_lambda, value_type=float),
                "ci_self_weight": ParameterValue(ci_self_weight, value_type=float),
                "unknown_motion_variance": ParameterValue(
                    unknown_motion_variance, value_type=float),
            }],
        ))
    actions.append(Node(
        package="hercules_localization_ros", executable="relative_observer_node",
        name="localization_relative_observer", output="screen",
        condition=IfCondition(PythonExpression(["'", localization_source, "' == 'estimate'"])),
        parameters=[{"observation_source": localization_observation_source,
                     "host_ip": airsim_host,
                     "drone_port": ParameterValue(drone_port, value_type=int),
                     "ugv_port": ParameterValue(ugv_port, value_type=int),
                     "uav_camera": "localization_front",
                     "uav_camera_fallback": "target_bottom",
                     "rate_hz": ParameterValue(localization_relative_rate, value_type=float),
                     "sensing_range": ParameterValue(localization_camera_range, value_type=float),
                     "range_std": 0.25, "bearing_std_rad": 0.017453292519943295}],
    ))
    actions.append(Node(
        package="hercules_tracking_ros", executable="target_observer_node",
        name="target_observer", output="screen", condition=distributed,
        parameters=[{"observation_source": observation_source,
                     "host_ip": airsim_host,
                     "rpc_port": ParameterValue(drone_port, value_type=int),
                     "drone_port": ParameterValue(drone_port, value_type=int),
                     "ugv_port": ParameterValue(ugv_port, value_type=int),
                     "tracking_rate": 4.0,
                     "tracking_measurement_std": 0.25,
                     "target_sensing_range": 100.0,
                     "truth_seed": ParameterValue(truth_seed, value_type=int),
                     "truth_rng_mode": truth_rng_mode}],
    ))
    for agent in VEHICLES[:-1]:
        actions.append(Node(
            package="hercules_tracking_ros", executable="tracker_node",
            namespace=f"hercules_tracking/{agent}", name="target_tracker",
            output="screen", condition=distributed,
            parameters=[{"agent_id": agent, "target_id": "Target1",
                         "tracking_window": 5.0,
                         "tracking_admm_rho": 1.0,
                         "tracking_admm_max_iterations": ParameterValue(tracking_iterations, value_type=int),
                         "maicp_enabled": ParameterValue(tracking_fixed_iterations, value_type=bool),
                         "maicp_class": "uav" if agent in UAVS else "ugv",
                         "maicp_margin": ParameterValue(PythonExpression([
                             "'", maicp_case, "' == 'tracking' and ",
                             maicp_margin_uav if agent in UAVS else maicp_margin_ugv, " or 0.0"]), value_type=float),
                         "maicp_covariance_gain": ParameterValue(
                             maicp_gain_uav if agent in UAVS else maicp_gain_ugv, value_type=float),
                         "tracking_admm_tolerance": 0.001,
                         "tracking_process_noise": 0.20,
                         "tracking_measurement_std": 0.25,
                         "round_timeout_sec": 0.030,
                         "seed_timeout_sec": 0.020,
                         "measurement_wait_sec": 0.010}],
        ))
    actions.append(Node(
        package="hercules_cbf_ros", executable="obstacle_observer_node",
        name="obstacle_observer", output="screen",
        condition=IfCondition(PythonExpression([
            "'", cbf_obstacle_source, "' == 'perception'"
        ])),
        parameters=[os.path.join(cbf_share, "config", "rural_perception.yaml"),
                    {"host_ip": airsim_host,
                     "rpc_port": ParameterValue(drone_port, value_type=int),
                     "car_port": ParameterValue(ugv_port, value_type=int)}],
    ))
    actions.append(Node(
        package="hercules_cbf_ros", executable="mission_collision_observer_node",
        name="mission_collision_observer", output="screen",
        condition=IfCondition(enable_collision_observer),
        parameters=[os.path.join(cbf_share, "config", "rural_perception.yaml"),
                    {"host_ip": airsim_host,
                     "rpc_port": ParameterValue(drone_port, value_type=int),
                     "car_port": ParameterValue(ugv_port, value_type=int)}],
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
            "video_rpc_host": airsim_host,
            "video_rpc_port": ParameterValue(video_rpc_port, value_type=int),
            "video_car_port": ParameterValue(video_car_port, value_type=int),
        }],
    ))
    return LaunchDescription(actions)
