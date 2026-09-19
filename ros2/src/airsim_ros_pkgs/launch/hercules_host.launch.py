"""Native Hero simulator, separate existing wrappers for drone and Husky RPC."""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {
        'vehicles': 'both', 'host_ip': os.environ.get('AIRSIM_HOST', '127.0.0.1'),
        'drone_port': os.environ.get('AIRSIM_MULTIROTOR_PORT', '41451'),
        'ugv_port': os.environ.get('AIRSIM_CAR_PORT', '41452'),
        'enable_api_control': 'false', 'benchmark_logging': 'false',
        # Keep renderer and clock policy explicit.  The wrapper reads these
        # parameters during initialization; hard-coding them here made a
        # host-side launch override silently ineffective.
        'is_vulkan': 'true', 'publish_clock': 'false', 'use_sim_time': 'false',
        'ugv_command_timeout_sec': '0.5',
        'update_airsim_control_every_n_sec': '0.05',
        'update_airsim_img_response_every_n_sec': '0.5',
        'update_lidar_every_n_sec': '0.5',
        'update_gpulidar_every_n_sec': '0.01',
        'update_echo_every_n_sec': '0.05',
    }
    actions = [DeclareLaunchArgument(k, default_value=v,
               **({'choices': ['both', 'drone', 'ugv']} if k == 'vehicles' else {}))
               for k, v in defaults.items()]
    for vehicle in ('drone', 'ugv'):
        parameters = {
            'host_ip': ParameterValue(LaunchConfiguration('host_ip'), value_type=str),
            'host_port': ParameterValue(LaunchConfiguration(vehicle + '_port'), value_type=int),
            'is_vulkan': ParameterValue(LaunchConfiguration('is_vulkan'), value_type=bool),
            'publish_clock': ParameterValue(LaunchConfiguration('publish_clock'), value_type=bool),
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'enable_object_transforms_list': False,
        }
        for key in ('enable_api_control', 'benchmark_logging'):
            parameters[key] = ParameterValue(LaunchConfiguration(key), value_type=bool)
        for key in defaults:
            if key.startswith('update_') or key == 'ugv_command_timeout_sec':
                parameters[key] = ParameterValue(LaunchConfiguration(key), value_type=float)
        actions.append(Node(
            package='airsim_ros_pkgs', executable='hercules_node',
            name='hercules_' + vehicle, output='screen', parameters=[parameters],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('vehicles'), "' in ('both', '", vehicle, "')"]))))
    return LaunchDescription(actions)
