cmake_minimum_required(VERSION 3.16)
project(hercules_localization VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)

add_library(${PROJECT_NAME}
  src/covariance.cpp
  src/math.cpp
  src/recursive_localization.cpp
  src/gs_ci.cpp
  src/global_ci.cpp
  src/localization.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE hercules_localization::
  DESTINATION lib/cmake/${PROJECT_NAME})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/hercules_localizationConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_localizationConfig.cmake
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_localizationConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_localizationConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/hercules_localizationConfigVersion.cmake
  DESTINATION lib/cmake/${PROJECT_NAME})

include(CTest)
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_localization test/test_localization.cpp)
  target_link_libraries(test_localization ${PROJECT_NAME})
  target_compile_options(test_localization PRIVATE -Wall -Wextra -Wpedantic)
endif()

ament_export_targets(hercules_localizationTargets HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_package()
