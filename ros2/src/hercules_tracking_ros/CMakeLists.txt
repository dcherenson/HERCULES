cmake_minimum_required(VERSION 3.16)
project(hercules_tracking_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(hercules_interfaces REQUIRED)
find_package(hercules_tracking REQUIRED)
find_package(rclcpp REQUIRED)

add_library(${PROJECT_NAME}_core
  src/neighbor_graph.cpp
  src/tracking_adapter.cpp
  src/tracking_protocol.cpp)
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME}_core PUBLIC
  Eigen3::Eigen hercules_tracking::hercules_tracking)
ament_target_dependencies(${PROJECT_NAME}_core PUBLIC hercules_interfaces)
target_compile_options(${PROJECT_NAME}_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(tracker_node src/tracker_node.cpp)
target_link_libraries(tracker_node ${PROJECT_NAME}_core)
ament_target_dependencies(tracker_node rclcpp)
target_compile_options(tracker_node PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS ${PROJECT_NAME}_core EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS tracker_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/target_observer_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME target_observer_node)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 hercules_interfaces hercules_tracking)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  add_executable(tracking_reference_dump test/tracking_reference_dump.cpp)
  target_link_libraries(tracking_reference_dump hercules_tracking::hercules_tracking)
  target_compile_options(tracking_reference_dump PRIVATE -Wall -Wextra -Wpedantic)
  ament_add_gtest(test_tracking_adapter test/test_tracking_adapter.cpp)
  target_link_libraries(test_tracking_adapter ${PROJECT_NAME}_core)
  target_compile_options(test_tracking_adapter PRIVATE -Wall -Wextra -Wpedantic)
  ament_add_gtest(test_tracking_protocol test/test_tracking_protocol.cpp)
  target_link_libraries(test_tracking_protocol ${PROJECT_NAME}_core)
  target_compile_options(test_tracking_protocol PRIVATE -Wall -Wextra -Wpedantic)
  ament_add_pytest_test(test_distributed_transport test/test_distributed_transport.py TIMEOUT 90
    ENV "TRACKER_NODE=$<TARGET_FILE:tracker_node>"
        "TRACKING_REFERENCE_DUMP=$<TARGET_FILE:tracking_reference_dump>"
        "HERCULES_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../../PythonClient/distributed_mission"
        "ROS_DOMAIN_ID=145" "ROS_LOCALHOST_ONLY=1")
  ament_add_pytest_test(test_target_observer test/test_target_observer.py TIMEOUT 60)
  ament_add_pytest_test(test_truth_observer_transport test/test_truth_observer_transport.py TIMEOUT 60
    ENV "TARGET_OBSERVER_NODE=${CMAKE_CURRENT_SOURCE_DIR}/scripts/target_observer_node.py"
        "ROS_DOMAIN_ID=147" "ROS_LOCALHOST_ONLY=1")
endif()

ament_package()
