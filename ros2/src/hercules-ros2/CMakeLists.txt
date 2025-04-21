cmake_minimum_required(VERSION 3.8)
project(hercules-ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
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

install(TARGETS
  imu_linear_interpolator_v2_node
  DESTINATION lib/${PROJECT_NAME}
)



ament_package()
