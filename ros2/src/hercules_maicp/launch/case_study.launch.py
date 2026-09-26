"""Select one independent paper case using the existing AirSim ROS stack."""
import math
import socket
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _launch(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    case = value('case')
    if case not in ('collision', 'tracking', 'localization'):
        raise ValueError('case must be collision, tracking, or localization')
    models = {'uav': [0.0] * 6, 'ugv': [0.0] * 6}
    if case == 'collision':
        if value('dynamics_file'):
            # Model fitting is a separate, pre-calibration operation.
            from hercules_maicp.dynamics import load_dynamics_models
            model = load_dynamics_models(value('dynamics_file'))
            models = {cls: entry.coefficients for cls, entry in model.items()}
        elif value('training').lower() != 'true':
            raise ValueError('collision calibration requires dynamics_file; use training:=true only for separate fitting missions')
    for cls in ('uav', 'ugv'):
        if len(models[cls]) != 6 or not all(math.isfinite(float(x)) for x in models[cls]):
            raise ValueError('frozen affine dynamics need six finite coefficients per class')
    arguments = {
        'dry_run': 'false', 'maicp_case': case, 'duration_sec': value('duration'),
        'maicp_steps': value('steps'), 'log_path': value('log_path'),
        'maicp_margin_uav': value('margin_uav'), 'maicp_margin_ugv': value('margin_ugv'),
        'maicp_model_uav': ' '.join(map(str, models['uav'])),
        'maicp_model_ugv': ' '.join(map(str, models['ugv'])),
        'maicp_gain_uav': value('gain_uav'), 'maicp_gain_ugv': value('gain_ugv'),
        'tracking_iterations': '50', 'tracking_fixed_iterations': 'true' if case == 'tracking' else 'false',
        'truth_seed': value('seed'), 'truth_rng_mode': 'seeded',
        'airsim_host': socket.gethostbyname(value('host')), 'startup_timeout_sec': '120.0',
        'enable_target': 'true' if case == 'tracking' else 'false',
        'target_source': 'distributed_tracking' if case == 'tracking' else 'truth',
        'target_observation_source': value('observation_source'),
        'localization_observation_source': value('observation_source'),
        'localization_source': 'estimate' if case == 'localization' else 'truth',
        'localization_algorithm': 'recursive_decentralized',
        'cbf_enabled': 'true' if case == 'collision' else 'false',
        'cbf_method': 'wang', 'cbf_obstacle_source': 'none',
        'enable_collision_observer': 'true', 'route_heading_rad': '0.0',
    }
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(str(
        Path(get_package_share_directory('hercules_mission_ros')) / 'launch/rural_nominal.launch.py')),
        launch_arguments=arguments.items())]


def generate_launch_description():
    defaults = dict(case='collision', margin_uav='0.2', margin_ugv='0.2',
                    steps='100', duration='10.0', seed='27', host='127.0.0.1',
                    log_path='/tmp/hercules_maicp/mission.jsonl', dynamics_file='',
                    training='false', observation_source='camera', gain_uav='0.20', gain_ugv='0.30')
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()] +
                             [OpaqueFunction(function=_launch)])
