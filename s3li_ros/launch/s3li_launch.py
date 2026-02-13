from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    calib_arg = DeclareLaunchArgument(
        'calib_file',
        default_value='camera_calibration_callab_dlrrmc-07-04.cal',
        description='Path to calde-style calibration file .cal'
    )

    calib_config = LaunchConfiguration('calib_file')

    calde_node = Node(
        package='s3li_ros',
        executable='calde_to_camerainfo_ros.py',
        name='calde_to_camerainfo',
        parameters=[{
            'calibration_file': calib_config,  # Questo è l'unico dinamico
            'left_namespace': 'g319c_left',
            'right_namespace': 'g319c_right'
        }],
        output='screen'
    )

    disparity_node = Node(
        package='s3li_ros',
        executable='stereo_disparity_node',
        name='stereo_disparity',
        parameters=[{
            'left_image_topic': '/g319c_left/image_raw',
            'right_image_topic': '/g319c_right/image_raw',
            'left_camera_info_topic': 'g319c_left/camera_info',
            'right_camera_info_topic': 'g319c_right/camera_info',
            'rescale': 0.25,
            'rescale_disp': 0.0,
            'visualize': False
        }],
        output='screen'
    )

    return LaunchDescription([
        calib_arg,
        calde_node,
        disparity_node
    ])
