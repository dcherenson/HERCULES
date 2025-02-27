cmake_minimum_required(VERSION 3.5)
project(octomap_ros)

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# Manually locate the OctoMap libraries and include directory
find_library(OCTOMAP_LIBRARY NAMES octomap
  HINTS /usr/local/lib /lib/x86_64-linux-gnu
)
if(NOT OCTOMAP_LIBRARY)
  message(FATAL_ERROR "Cannot find octomap library")
endif()

find_library(OCTOMATH_LIBRARY NAMES octomath
  HINTS /usr/local/lib /lib/x86_64-linux-gnu
)
if(NOT OCTOMATH_LIBRARY)
  message(FATAL_ERROR "Cannot find octomath library")
endif()

find_path(OCTOMAP_INCLUDE_DIR NAMES octomap/octomap.h
  HINTS /usr/local/include /usr/include
)
if(NOT OCTOMAP_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot find octomap include directory")
endif()

message(STATUS "Found octomap library: ${OCTOMAP_LIBRARY}")
message(STATUS "Found octomath library: ${OCTOMATH_LIBRARY}")
message(STATUS "Found octomap include dir: ${OCTOMAP_INCLUDE_DIR}")

# Create imported targets for octomap and octomath
add_library(octomap_lib SHARED IMPORTED)
set_target_properties(octomap_lib PROPERTIES
  IMPORTED_LOCATION "${OCTOMAP_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${OCTOMAP_INCLUDE_DIR}"
)

add_library(octomath_lib SHARED IMPORTED)
set_target_properties(octomath_lib PROPERTIES
  IMPORTED_LOCATION "${OCTOMATH_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${OCTOMAP_INCLUDE_DIR}"
)

# Build the octomap_ros library
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/conversions.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${OCTOMAP_INCLUDE_DIR}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_${PROJECT_NAME}
    test/test_conversions.cpp
  )
  target_link_libraries(test_${PROJECT_NAME}
    gtest_main
    octomap_lib
    octomath_lib
    ${PROJECT_NAME}
  )
endif()

ament_auto_package()
