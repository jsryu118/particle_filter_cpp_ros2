import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from launch.actions import TimerAction

def generate_launch_description():
    # Define the path to the config.yaml file
    config_dir = os.path.join(get_package_share_directory('particle_filter'), 'config')
    map_dir = os.path.join(get_package_share_directory('particle_filter'), 'maps')
    config_file = os.path.join(config_dir, 'config.yaml')
    map_file = os.path.join(map_dir, 'map_data.yaml')

    # Map server node
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_file}],
        remappings=[('/map', '/pf_map')]
    )

    # Particle filter node
    particle_filter_node = Node(
        package='particle_filter',
        executable='particle_filter_node',
        name='particle_filter',
        output='screen',
        parameters=[config_file]
    )

    configure_map_server = TimerAction(
        period=3.0,  # 1초 대기
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'lifecycle', 'set', '/map_server', 'configure'],
                output='screen'
            )
        ]
    )
    # Return the launch description
    return LaunchDescription([
        map_server_node,
        particle_filter_node,
        # configure_map_server
    ])
