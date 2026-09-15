# hercules_cbf

`hercules_cbf` is a ROS-independent C++17 implementation of the executable
Python distributed CBF behavior. It exposes ordered constraint construction,
the native OSQP wrapper, Python-compatible post-solve clipping, Mestres local
projection, Wang neighbor responsibility splitting, and the documented
fail-safe paths. ROS conversion and mission scheduling live in
`hercules_cbf_ros`.

Build with CMake or colcon after the Docker image installs the pinned OSQP and
QDLDL sources. `ctest` runs the focused native regressions and the 12-case
Python/C++ row and control parity harness.
