import os 
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def check_arguments(context, *args, **kwargs):
    calib_file = LaunchConfiguration('calib_file').perform(context)
    
    if not calib_file:
        raise Exception("Launch error: 'calib_file' argument is mandatory but was not provided.")
    return

def generate_launch_description():

    ## *.cal files are recommended, as they provide more precise stereo calibration
    ## camera_calibration_callab_dlrrmc-07-04.cal -> capo_grillo, capo_grillo_2
    ## camera_calibration_callab_dlrrmc-07-05.cal -> vegetation, waterfront
    ## camera_calibration_callab_dlrrmc-07-06.cal -> straight_path, moon_lake and lava_tunnel
    calib_arg = DeclareLaunchArgument(
        'calib_file',
        description="""
        Path to calde-style calibration file .cal (MANDATORY!)
        Recommendations: use the following calibration files for the associated sequences listed here
        camera_calibration_callab_dlrrmc-07-04.cal -> capo_grillo, capo_grillo_2
        camera_calibration_callab_dlrrmc-07-05.cal -> vegetation, waterfront
        camera_calibration_callab_dlrrmc-07-06.cal -> straight_path, moon_lake and lava_tunnel
        """
    )

    calib_config = LaunchConfiguration('calib_file')

    calde_node = Node(
        package='s3li_ros',
        executable='calde_to_camerainfo_ros.py',
        name='calde_to_camerainfo',
        parameters=[{
            'calibration_file': calib_config, 
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
            'visualize': True
        }],
        output='screen'
    )

    tf_left_camera_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_left_camera_lidar_publisher',
        arguments=[
            '0.15', '-0.02', '-0.02', 
            '-0.005', '0.0', '1.623', 'g319c_left', 'lidar']
    )

    tf_imu_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_left_camera_lidar_publisher',
        arguments=[
            '0.00749803', '0.25481334', '0.00961434',
            '-0.50047531', '0.50210171', '-0.50009389', '0.49731723',
            'imu', 'g319c_left']
    )

    # This relates to the "un-rectified", therefore physical, camera pair
    tf_stereo_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='stereo_link_publisher',
        arguments=[
            '0.08914264', '0.00009292', '-0.00204204', 
            '-0.00183492', '0.01381102', '-0.00119056', '0.99989932', 
            'g319c_left', 'g319c_right'
        ]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('s3li_ros'), 'cfg', 's3li.rviz')],
        output='screen'
    )

    return LaunchDescription([
        calib_arg,
        OpaqueFunction(function=check_arguments),
        calde_node,
        disparity_node,
        tf_imu_camera, 
        tf_left_camera_lidar, 
        tf_stereo_camera,
        rviz_node
    ])
