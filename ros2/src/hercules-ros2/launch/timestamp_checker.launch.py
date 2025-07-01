from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    topic_arg = DeclareLaunchArgument(
        'topic',
        default_value='/hercules_node/Husky1/ground_truth/odom_local',
        description='The topic to check timestamps for')

    type_arg = DeclareLaunchArgument(
        'message_type',
        default_value='nav_msgs/msg/Odometry',
        description='The message type: sensor_msgs/msg/Imu or nav_msgs/msg/Odometry')

    period_arg = DeclareLaunchArgument(
        'expected_period',
        default_value='0.0025',
        description='Expected time between messages (seconds)')

    tol_arg = DeclareLaunchArgument(
        'tolerance',
        default_value='0.0005',
        description='Allowed deviation from expected_period')

    checker_node = Node(
        package='hercules-ros2',
        executable='bag_timestamp_checker_node',
        name='bag_timestamp_checker_node',
        output='screen',
        parameters=[{
            'topic': LaunchConfiguration('topic'),
            'message_type': LaunchConfiguration('message_type'),
            'expected_period': LaunchConfiguration('expected_period'),
            'tolerance': LaunchConfiguration('tolerance'),
        }]
    )

    return LaunchDescription([
        topic_arg,
        type_arg,
        period_arg,
        tol_arg,
        checker_node
    ])
