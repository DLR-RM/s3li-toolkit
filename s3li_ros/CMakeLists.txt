cmake_minimum_required(VERSION 3.15)
project(s3li_ros)

## Use C++14
add_compile_options(-std=c++17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(stereo_disparity_node
  src/stereo_disparity_node.cpp
)

set(ROS2_LIBRARIES
    "visualization_msgs"
    "message_filters"
    "std_msgs"
    "pcl_ros"
    "image_geometry"
    "rclcpp"
    "tf2_ros"
    "sensor_msgs"
    "geometry_msgs"
    "cv_bridge")
ament_target_dependencies(stereo_disparity_node ${ROS2_LIBRARIES})

target_link_libraries(stereo_disparity_node
  ${OpenCV_LIBRARIES}
  "stdc++fs"
)

install(TARGETS stereo_disparity_node
  	DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS s3li_ros/calde_to_camerainfo_ros.py
	DESTINATION lib/${PROJECT_NAME}
)

# Python install (calls setup.py)
ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
