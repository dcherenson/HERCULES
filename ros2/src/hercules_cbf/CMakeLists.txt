cmake_minimum_required(VERSION 3.16)
project(hercules_cbf VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(osqp CONFIG REQUIRED)

add_library(${PROJECT_NAME} src/constraints.cpp src/filter.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen osqp::osqp)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE hercules_cbf:: DESTINATION lib/cmake/${PROJECT_NAME})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/hercules_cbfConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_cbfConfig.cmake
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_cbfConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_cbfConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_cbfConfigVersion.cmake
  DESTINATION lib/cmake/${PROJECT_NAME})

include(CTest)
if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  add_executable(test_cbf test/test_cbf.cpp)
  target_link_libraries(test_cbf ${PROJECT_NAME} GTest::gtest_main)
  target_compile_options(test_cbf PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME test_cbf COMMAND test_cbf)

  add_executable(cbf_parity_dump test/cbf_parity_dump.cpp)
  target_link_libraries(cbf_parity_dump ${PROJECT_NAME})
  target_compile_options(cbf_parity_dump PRIVATE -Wall -Wextra -Wpedantic)

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME test_python_cpp_cbf_parity
    COMMAND ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_SOURCE_DIR}/test/test_python_cpp_cbf_parity.py)
  set_tests_properties(test_python_cpp_cbf_parity PROPERTIES
    ENVIRONMENT "CBF_PARITY_EXE=$<TARGET_FILE:cbf_parity_dump>;HERCULES_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/../../../PythonClient/distributed_mission")
endif()
