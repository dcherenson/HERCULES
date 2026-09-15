"""Launch the eight independent RuralAustralia Target1 trackers."""

from launch import LaunchDescription
from launch_ros.actions import Node


AGENTS = [
    "Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5",
    "Husky1", "Husky2", "Husky3",
]


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="hercules_tracking_ros",
            executable="tracker_node",
            namespace=f"hercules_tracking/{agent}",
            name="target_tracker",
            output="screen",
            parameters=[{"agent_id": agent}],
        )
        for agent in AGENTS
    ])
