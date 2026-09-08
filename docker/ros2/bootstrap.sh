#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
[[ -f /.dockerenv ]] || { echo 'Run bootstrap inside the development container.' >&2; exit 2; }
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT
if [[ ! -f "$REPO_ROOT/external/rpclib/rpclib-2.3.0/CMakeLists.txt" ]]; then
  [[ ! -e "$REPO_ROOT/external/rpclib/rpclib-2.3.0" ]] || { echo 'Incomplete rpclib directory; refusing to overwrite it.' >&2; exit 1; }
  curl -fL --retry 3 https://github.com/rpclib/rpclib/archive/refs/tags/v2.3.0.zip -o "$tmp/rpc.zip"
  unzip -q "$tmp/rpc.zip" -d "$tmp"
  mkdir -p "$REPO_ROOT/external/rpclib"
  mv "$tmp/rpclib-2.3.0" "$REPO_ROOT/external/rpclib/"
fi
if [[ ! -f "$REPO_ROOT/AirLib/deps/eigen3/Eigen/Core" ]]; then
  [[ ! -e "$REPO_ROOT/AirLib/deps/eigen3" ]] || { echo 'Incomplete Eigen directory; refusing to overwrite it.' >&2; exit 1; }
  curl -fL --retry 3 https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip -o "$tmp/eigen.zip"
  unzip -q "$tmp/eigen.zip" -d "$tmp"
  mkdir -p "$REPO_ROOT/AirLib/deps/eigen3"
  mv "$tmp/eigen-3.4.0/Eigen" "$REPO_ROOT/AirLib/deps/eigen3/"
fi
