from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    imu_in_arg = DeclareLaunchArgument(
        'imu_input_topic', default_value='/hercules_node/imu/data',
        description='Raw IMU topic name')
    imu_out_arg = DeclareLaunchArgument(
        'imu_output_topic', default_value='/vins/imu',
        description='Downsampled IMU topic name')
    imu_rate_arg = DeclareLaunchArgument(
        'imu_rate_hz', default_value='200.0',
        description='Target IMU rate (Hz)')

    cam_in_arg = DeclareLaunchArgument(
        'cam_input_topic', default_value='/hercules_node/cam0/image_raw',
        description='Raw RGB camera topic name')
    cam_out_arg = DeclareLaunchArgument(
        'cam_output_topic', default_value='/vins/camera_gray',
        description='Downsampled & grayscale camera topic name')
    cam_rate_arg = DeclareLaunchArgument(
        'cam_rate_hz', default_value='20.0',
        description='Target camera rate (Hz)')

    downs_node = Node(
        package='your_package_name',            # replace with your package
        executable='downsync_node',             # the node built above
        name='downsync_node',
        output='screen',
        parameters=[{
          'imu_input_topic':    LaunchConfiguration('imu_input_topic'),
          'imu_output_topic':   LaunchConfiguration('imu_output_topic'),
          'imu_rate_hz':        LaunchConfiguration('imu_rate_hz'),
          'cam_input_topic':    LaunchConfiguration('cam_input_topic'),
          'cam_output_topic':   LaunchConfiguration('cam_output_topic'),
          'cam_rate_hz':        LaunchConfiguration('cam_rate_hz'),
        }]
    )

    return LaunchDescription([
        imu_in_arg, imu_out_arg, imu_rate_arg,
        cam_in_arg, cam_out_arg, cam_rate_arg,
        downs_node
    ])
