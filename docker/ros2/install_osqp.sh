#!/usr/bin/env bash
set -euo pipefail

build_root=/tmp/hercules-osqp-build
osqp_source="$build_root/osqp"
qdldl_source="$build_root/qdldl"
rm -rf "$build_root"
mkdir -p "$build_root"
git clone --quiet https://github.com/osqp/osqp.git "$osqp_source"
git -C "$osqp_source" fetch --quiet --depth 1 origin 236713ce9a56c182ac3230d52108f952afce1523
git -C "$osqp_source" checkout --quiet --detach 236713ce9a56c182ac3230d52108f952afce1523
test "$(git -C "$osqp_source" rev-parse HEAD)" = 236713ce9a56c182ac3230d52108f952afce1523
git clone --quiet https://github.com/osqp/qdldl.git "$qdldl_source"
git -C "$qdldl_source" fetch --quiet --depth 1 origin 138fdac58b9cd1c4137ff1b99152c8108a6cff5b
git -C "$qdldl_source" checkout --quiet --detach 138fdac58b9cd1c4137ff1b99152c8108a6cff5b
test "$(git -C "$qdldl_source" rev-parse HEAD)" = 138fdac58b9cd1c4137ff1b99152c8108a6cff5b
cmake -S "$osqp_source" -B "$build_root/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DFETCHCONTENT_SOURCE_DIR_QDLDL="$qdldl_source" \
  -DOSQP_BUILD_SHARED_LIB=ON -DOSQP_BUILD_STATIC_LIB=OFF \
  -DOSQP_BUILD_UNITTESTS=OFF -DOSQP_BUILD_DEMO_EXE=OFF \
  -DOSQP_USE_FLOAT=OFF -DOSQP_USE_LONG=OFF \
  -DOSQP_ENABLE_DERIVATIVES=OFF -DOSQP_CODEGEN=OFF \
  -DOSQP_ENABLE_PRINTING=OFF -DOSQP_ENABLE_PROFILING=OFF
cmake --build "$build_root/build" --parallel 2
cmake --install "$build_root/build"
test -f /usr/local/lib/cmake/osqp/osqp-config.cmake
rm -rf "$build_root"
