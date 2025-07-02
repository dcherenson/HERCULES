from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    imu_in_arg = DeclareLaunchArgument(
        'imu_input_topic', default_value='/hercules_node/Husky1/imu/imu',
        description='Raw IMU topic name')
    imu_out_arg = DeclareLaunchArgument(
        'imu_output_topic', default_value='/VINS/Husky1/imu',
        description='Downsampled IMU topic name')
    imu_rate_arg = DeclareLaunchArgument(
        'imu_rate_hz', default_value='150.0',
        description='Target IMU rate (Hz)')

    cam_in_arg = DeclareLaunchArgument(
        'cam_input_topic', default_value='/hercules_node/Husky1/front_center_Scene/image',
        description='Raw RGB camera topic name')
    cam_out_arg = DeclareLaunchArgument(
        'cam_output_topic', default_value='/VINS/Husky1/front_center_Scene/image_greyscale',
        description='Downsampled & grayscale camera topic name')
    cam_rate_arg = DeclareLaunchArgument(
        'cam_rate_hz', default_value='20.0',
        description='Target camera rate (Hz)')

    downs_node = Node(
        package='hercules-ros2',            
        executable='downsample_synchronizer_node',             # the node built above
        name='downsample_synchronizer_node',
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
