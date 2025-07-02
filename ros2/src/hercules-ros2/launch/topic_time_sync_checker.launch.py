from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Launch arguments
    # camera_arg = DeclareLaunchArgument(
    #     'camera_topic',
    #     default_value='/hercules_node/Husky1/front_center_Scene/image',
    #     description='Name of the camera topic (sensor_msgs/msg/Image)'
    # )
    # imu_arg = DeclareLaunchArgument(
    #     'imu_topic',
    #     default_value='/hercules_node/Husky1/imu/imu',
    #     description='Name of the IMU topic (sensor_msgs/msg/Imu)'
    # )

    camera_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/VINS/Husky1/front_center_Scene/image_greyscale',
        description='Name of the camera topic (sensor_msgs/msg/Image)'
    )
    imu_arg = DeclareLaunchArgument(
        'imu_topic',
        default_value='/VINS/Husky1/imu',
        description='Name of the IMU topic (sensor_msgs/msg/Imu)'
    )

    tol_arg = DeclareLaunchArgument(
        'sync_tolerance',
        default_value='0.002',
        description='Max allowed time difference (seconds) between camera & IMU'
    )
    buf_arg = DeclareLaunchArgument(
        'imu_buffer_size',
        default_value='2000',
        description='How many IMU stamps to keep in buffer for matching'
    )

    # Node definition
    checker_node = Node(
        package='hercules-ros2',            
        executable='topic_time_sync_checker_node',   # your C++ node executable
        name='topic_time_sync_checker_node',
        output='screen',
        parameters=[{
            'camera_topic': LaunchConfiguration('camera_topic'),
            'imu_topic':    LaunchConfiguration('imu_topic'),
            'sync_tolerance': LaunchConfiguration('sync_tolerance'),
            'imu_buffer_size': LaunchConfiguration('imu_buffer_size'),
        }]
    )

    return LaunchDescription([
        camera_arg,
        imu_arg,
        tol_arg,
        buf_arg,
        checker_node
    ])
