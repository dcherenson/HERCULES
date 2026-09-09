cmake_minimum_required(VERSION 3.16)
project(hercules_mission_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(hercules_interfaces REQUIRED)
find_package(hercules_mission_core REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)

add_library(${PROJECT_NAME}_core
  src/state_adapter.cpp
  src/state_cache.cpp)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_core PUBLIC
  Eigen3::Eigen hercules_mission_core::hercules_mission_core)

add_library(${PROJECT_NAME}_ros src/ros_state_adapter.cpp)
target_include_directories(${PROJECT_NAME}_ros PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_ros PUBLIC ${PROJECT_NAME}_core)
ament_target_dependencies(${PROJECT_NAME}_ros PUBLIC
  hercules_interfaces nav_msgs)

add_executable(state_node src/state_node.cpp)
target_link_libraries(state_node ${PROJECT_NAME}_ros)
ament_target_dependencies(state_node rclcpp)

foreach(target ${PROJECT_NAME}_core ${PROJECT_NAME}_ros state_node)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()

install(TARGETS ${PROJECT_NAME}_core ${PROJECT_NAME}_ros
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib)
install(TARGETS state_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 hercules_interfaces hercules_mission_core nav_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_state_adapter test/test_state_adapter.cpp)
  target_link_libraries(test_state_adapter ${PROJECT_NAME}_core)
  target_compile_options(test_state_adapter PRIVATE -Wall -Wextra -Wpedantic)

  ament_add_gtest(test_ros_state_adapter test/test_ros_state_adapter.cpp)
  target_link_libraries(test_ros_state_adapter ${PROJECT_NAME}_ros)
  target_compile_options(test_ros_state_adapter PRIVATE -Wall -Wextra -Wpedantic)

  ament_add_pytest_test(test_state_node test/test_state_node.py TIMEOUT 60
    ENV "MISSION_STATE_NODE=$<TARGET_FILE:state_node>"
        "ROS_DOMAIN_ID=144" "ROS_LOCALHOST_ONLY=1")
endif()

ament_package()
