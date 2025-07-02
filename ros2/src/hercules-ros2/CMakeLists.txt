cmake_minimum_required(VERSION 3.8)
project(hercules-ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(airsim_interfaces REQUIRED)
find_package(OpenCV REQUIRED)

# find_package(airsim_ros_pkgs REQUIRED)

# TO BUILD WITH CUSTOM MESSAGES
# find_package(rosidl_default_generators REQUIRED)

# rosidl_generate_interfaces(${PROJECT_NAME}
#   "msg/YourCustomMessage.msg"
#   DEPENDENCIES std_msgs geometry_msgs
# )

# ament_export_dependencies(rosidl_default_runtime)

# EXAMPLE OF HOW YOU WOULD CREATE NEW CPP ROS2 EXECUTABLE
# add_executable(airsim_node src/airsim_node.cpp)
# target_link_libraries(airsim_node airsim_ros AirLib)
# ament_target_dependencies(airsim_node 
#   rclcpp
# )


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


add_executable(imu_linear_interpolator_node src/imu_linear_interpolator.cpp)
ament_target_dependencies(imu_linear_interpolator_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
)

install(TARGETS
  imu_linear_interpolator_node
  DESTINATION lib/${PROJECT_NAME}
)


add_executable(imu_linear_interpolator_v2_node src/imu_linear_interpolator_v2.cpp)
ament_target_dependencies(imu_linear_interpolator_v2_node
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


install(TARGETS
  imu_linear_interpolator_v2_node
  DESTINATION lib/${PROJECT_NAME}
)


# depth viewing node______________
add_executable(view_depthperspective_pointcloud_node
 src/view_depthperspective_pointcloud.cpp)
ament_target_dependencies(view_depthperspective_pointcloud_node

  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  cv_bridge
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


install(TARGETS
  view_depthperspective_pointcloud_node
  DESTINATION lib/${PROJECT_NAME}
)

# thermal image node___________________
add_executable(thermal_image_node
 src/thermal_image_node.cpp)
ament_target_dependencies(thermal_image_node

  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  cv_bridge
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


install(TARGETS
thermal_image_node
  DESTINATION lib/${PROJECT_NAME}
)
# ____________________________
# bag timestamp checker node______
add_executable(bag_timestamp_checker_node
 src/bag_timestamp_checker.cpp)
 ament_target_dependencies(bag_timestamp_checker_node
 rclcpp
 std_msgs
 sensor_msgs
 geometry_msgs
 nav_msgs
 tf2
 tf2_ros
 tf2_sensor_msgs
 cv_bridge
 airsim_interfaces
)


install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


install(TARGETS
bag_timestamp_checker_node
  DESTINATION lib/${PROJECT_NAME}
)
# ________________________________
# topic time sync checker node__________________
add_executable(topic_time_sync_checker_node
 src/topic_time_sync_checker.cpp)
 ament_target_dependencies(topic_time_sync_checker_node
 rclcpp
 std_msgs
 sensor_msgs
 geometry_msgs
 nav_msgs
 tf2
 tf2_ros
 tf2_sensor_msgs
 cv_bridge
 airsim_interfaces
)


install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


install(TARGETS
topic_time_sync_checker_node
  DESTINATION lib/${PROJECT_NAME}
)
#_______________________________________________
# imu republisher node_________________________
add_executable(imu_republisher_node
 src/imu_republisher_node.cpp)
 ament_target_dependencies(imu_republisher_node
 rclcpp
 std_msgs
 sensor_msgs
 geometry_msgs
 nav_msgs
 tf2
 tf2_ros
 tf2_sensor_msgs
 cv_bridge
 airsim_interfaces
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


install(TARGETS
imu_republisher_node
  DESTINATION lib/${PROJECT_NAME}
)
# _____________________________________________
# downsample and synchronize topics node_______
add_executable(downsample_synchronizer_node
 src/downsample_synchronizer_node.cpp)
 ament_target_dependencies(downsample_synchronizer_node
 rclcpp
 std_msgs
 sensor_msgs
 geometry_msgs
 nav_msgs
 tf2
 tf2_ros
 tf2_sensor_msgs
 cv_bridge
 airsim_interfaces
 cv_bridge 
 OpenCV 
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)


install(TARGETS
downsample_synchronizer_node
  DESTINATION lib/${PROJECT_NAME}
)

# _____________________________________________

# Install Python scripts
install(PROGRAMS
  scripts/test_depthperspective.py 
  scripts/depthvis.py 
  scripts/multi_robot_topic_monitor.py 
  scripts/full_topic_echo.py 
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
