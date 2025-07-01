from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    topic_arg = DeclareLaunchArgument(
        'topic',
        default_value='/hercules_node/Drone1/imu/imu',
        description='The topic to check timestamps for')

    period_arg = DeclareLaunchArgument(
        'expected_period',
        default_value='0.05',
        description='Expected time between messages (seconds)')

    tol_arg = DeclareLaunchArgument(
        'tolerance',
        default_value='0.0001',
        description='Allowed deviation from expected_period')

    checker_node = Node(
        package='hercules-ros2',
        executable='bag_timestamp_checker_node',
        name='bag_timestamp_checker_node',
        output='screen',
        parameters=[{
            'topic': LaunchConfiguration('topic'),
            'expected_period': LaunchConfiguration('expected_period'),
            'tolerance': LaunchConfiguration('tolerance'),
        }]
    )

    return LaunchDescription([
        topic_arg,
        period_arg,
        tol_arg,
        checker_node
    ])
