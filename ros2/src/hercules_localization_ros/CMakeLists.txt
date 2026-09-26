cmake_minimum_required(VERSION 3.16)
project(hercules_localization_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(airsim_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(hercules_interfaces REQUIRED)
find_package(hercules_localization REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(${PROJECT_NAME}_core src/localization_adapter.cpp)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(${PROJECT_NAME}_core PUBLIC hercules_interfaces)
target_link_libraries(${PROJECT_NAME}_core PUBLIC Eigen3::Eigen hercules_localization::hercules_localization)
target_compile_options(${PROJECT_NAME}_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(localization_node src/localization_node.cpp)
target_link_libraries(localization_node PRIVATE ${PROJECT_NAME}_core)
ament_target_dependencies(localization_node PUBLIC airsim_interfaces geometry_msgs hercules_interfaces nav_msgs rclcpp sensor_msgs)
target_compile_options(localization_node PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS ${PROJECT_NAME}_core EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(TARGETS localization_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/relative_observer_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME relative_observer_node)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 airsim_interfaces geometry_msgs hercules_interfaces hercules_localization nav_msgs rclcpp sensor_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_localization_adapter test/test_localization_adapter.cpp)
  target_link_libraries(test_localization_adapter ${PROJECT_NAME}_core)
  target_compile_options(test_localization_adapter PRIVATE -Wall -Wextra -Wpedantic)

  ament_add_pytest_test(test_localization_transport test/test_localization_transport.py TIMEOUT 90
    ENV "LOCALIZATION_NODE=$<TARGET_FILE:localization_node>"
        "ROS_DOMAIN_ID=149" "ROS_LOCALHOST_ONLY=1")
endif()

ament_package()
