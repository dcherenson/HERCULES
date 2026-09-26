"""One private calibration store and pinball/flooding participant per robot."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='hercules_maicp', executable='calibration_worker',
             name='maicp_' + agent, output='screen', parameters=[{'agent_id': agent}])
        for agent in ['Drone1', 'Drone2', 'SimpleFlight', 'Husky1', 'Husky2', 'Husky3']
    ])
