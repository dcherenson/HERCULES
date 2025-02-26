#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Set the full path to your octomap (.bt) file.
    octree_file = '/home/sgarimella34/Downloads/Ausland1.bt'
    # octree_file = '/home/sgarimella34/Downloads/sample_octomap.bt'

    # Launch the octomap_server node.
    octomap_server_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        parameters=[{'octree_file': octree_file}]
    )

    # Optionally, specify an RViz configuration file.
    # If you have a config file with an Octomap display already set up, provide its path.
    rviz_config_file = os.path.join(
        os.path.expanduser('~'),
        'rviz_config',  # change this to your directory
        'octomap.rviz'  # change this to your config filename
    )

    # Launch rviz2 (if the config file exists, pass it; otherwise, launch without a config).
    if os.path.exists(rviz_config_file):
        rviz2_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        )
    else:
        rviz2_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen'
        )

    return LaunchDescription([
        octomap_server_node,
        rviz2_node,
    ])
