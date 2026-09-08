cmake_minimum_required(VERSION 3.16)
project(hercules_tracking)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}
  src/models.cpp
  src/linear_algebra.cpp
  src/target_measurement.cpp
  src/target_track.cpp
  src/target_tracker.cpp
  src/synchronous_network.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)

  ament_add_gtest(test_tracking test/test_tracking.cpp)
  target_link_libraries(test_tracking ${PROJECT_NAME})
  target_compile_options(test_tracking PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(tracking_parity_dump test/tracking_parity_dump.cpp)
  target_link_libraries(tracking_parity_dump ${PROJECT_NAME})
  target_compile_options(tracking_parity_dump PRIVATE -Wall -Wextra -Wpedantic)

  ament_add_pytest_test(test_python_cpp_parity test/test_python_cpp_parity.py
    TIMEOUT 120
    ENV "TRACKING_PARITY_EXE=$<TARGET_FILE:tracking_parity_dump>"
        "HERCULES_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../../PythonClient/distributed_mission")
endif()

ament_package()
