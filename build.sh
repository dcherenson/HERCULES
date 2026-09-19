#!/usr/bin/env bash

# get path of current script: https://stackoverflow.com/a/39340259/207661
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$SCRIPT_DIR"  >/dev/null

set -e
set -x

# debug=true
debug=false
gcc=false
# Parse command line arguments
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
    --debug)
        debug=true
        shift # past argument
        ;;
    --gcc)
        gcc=true
        shift # past argument
        ;;
    esac

done

function version_less_than_equal_to() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" = "$1"; }

# check for rpclib
RPC_VERSION_FOLDER="rpclib-2.3.0"
if [ ! -d "./external/rpclib/$RPC_VERSION_FOLDER" ]; then
    echo "ERROR: new version of AirSim requires newer rpclib."
    echo "please run setup.sh first and then run build.sh again."
    exit 1
fi

# check for local cmake build created by setup.sh
if [ -d "./cmake_build" ]; then
    if [ "$(uname)" == "Darwin" ]; then
        CMAKE="$(greadlink -f cmake_build/bin/cmake)"
    else
        CMAKE="$(readlink -f cmake_build/bin/cmake)"
    fi
else
    CMAKE=$(which cmake)
fi

# variable for build output
if $debug; then
    build_dir=build_debug
else
    build_dir=build_release
fi 
if [ "$(uname)" == "Darwin" ]; then
    command -v brew >/dev/null 2>&1 || {
        echo "ERROR: Homebrew is required for the macOS AirSim toolchain." >&2
        exit 1
    }

    # setup.sh installs llvm@18, which is keg-only. Prefer that exact
    # toolchain, then fall back to unversioned llvm for existing checkouts.
    # Callers can override the keg with HERCULES_LLVM_PREFIX or CC/CXX.
    if [[ -z "${CC:-}" || -z "${CXX:-}" ]]; then
        LLVM_PREFIX="${HERCULES_LLVM_PREFIX:-}"
        if [[ -z "$LLVM_PREFIX" ]]; then
            if LLVM_PREFIX="$(brew --prefix llvm@18 2>/dev/null)"; then
                :
            elif LLVM_PREFIX="$(brew --prefix llvm 2>/dev/null)"; then
                :
            else
                LLVM_PREFIX=''
            fi
        fi

        if [[ -n "$LLVM_PREFIX" && -x "$LLVM_PREFIX/bin/clang" && -x "$LLVM_PREFIX/bin/clang++" ]]; then
            export CC="${CC:-$LLVM_PREFIX/bin/clang}"
            export CXX="${CXX:-$LLVM_PREFIX/bin/clang++}"
        else
            export CC="${CC:-$(command -v clang || true)}"
            export CXX="${CXX:-$(command -v clang++ || true)}"
        fi
    fi

    # CMake's compiler cache is more reliable with absolute paths, while
    # callers may naturally provide CC=clang/CXX=clang++ on the command line.
    if [[ "$CC" != */* ]]; then
        CC="$(command -v "$CC" 2>/dev/null || true)"
    fi
    if [[ "$CXX" != */* ]]; then
        CXX="$(command -v "$CXX" 2>/dev/null || true)"
    fi
    export CC CXX
    [[ -x "$CC" && -x "$CXX" ]] || {
        echo "ERROR: could not find usable macOS C/C++ compilers (CC=$CC CXX=$CXX)." >&2
        echo "Install llvm@18 with setup.sh or set HERCULES_LLVM_PREFIX, CC, and CXX." >&2
        exit 1
    }
else
    VERSION=$(lsb_release -rs | cut -d. -f1)
    if $gcc; then
        export CC="gcc-12"
        export CXX="g++-12"
    else
        export CC="clang-12"
        export CXX="clang++-12"
    fi
fi

#install EIGEN library
if [[ ! -d "./AirLib/deps/eigen3/Eigen" ]]; then
    echo "### Eigen is not installed. Please run setup.sh first."
    exit 1
fi

echo "putting build in $build_dir folder, to clean, just delete the directory..."

# this ensures the cmake files will be built in our $build_dir instead.
if [[ -f "./cmake/CMakeCache.txt" ]]; then
    rm "./cmake/CMakeCache.txt"
fi
if [[ -d "./cmake/CMakeFiles" ]]; then
    rm -rf "./cmake/CMakeFiles"
fi



if [[ ! -d $build_dir ]]; then
    mkdir -p $build_dir
fi

# Keep native Apple Silicon builds arm64. The old x86_64
# CMAKE_APPLE_SILICON_PROCESSOR override caused CMake/Unreal builds to drift
# into Rosetta; callers may explicitly select another architecture with
# HERCULES_MAC_ARCH when they intentionally need a non-native build.
CMAKE_ARGS=()
MAC_ARCH=''
if [ "$(uname)" == "Darwin" ]; then
    MAC_ARCH="${HERCULES_MAC_ARCH:-$(uname -m)}"
    case "$MAC_ARCH" in
        arm64|x86_64)
            ;;
        *)
            echo "ERROR: unsupported HERCULES_MAC_ARCH: $MAC_ARCH (expected arm64 or x86_64)." >&2
            exit 1
            ;;
    esac
    CMAKE_ARGS+=("-DCMAKE_OSX_ARCHITECTURES=$MAC_ARCH")
    CMAKE_ARGS+=("-DCMAKE_C_COMPILER=$CC")
    CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=$CXX")
fi

pushd $build_dir  >/dev/null
if $debug; then
    folder_name="Debug"
    "$CMAKE" ../cmake -DCMAKE_BUILD_TYPE=Debug "${CMAKE_ARGS[@]}" \
        || (popd && rm -r $build_dir && exit 1)   
else
    folder_name="Release"
    "$CMAKE" ../cmake -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARGS[@]}" \
        || (popd && rm -r $build_dir && exit 1)
fi
popd >/dev/null


pushd $build_dir  >/dev/null
# final linking of the binaries can fail due to a missing libc++abi library
# (happens on Fedora, see https://bugzilla.redhat.com/show_bug.cgi?id=1332306).
# So we only build the libraries here for now
if [[ -n "${BUILD_JOBS:-}" ]]; then
    BUILD_JOBS_VALUE="$BUILD_JOBS"
elif command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS_VALUE="$(nproc)"
elif [ "$(uname)" == "Darwin" ] && command -v sysctl >/dev/null 2>&1; then
    BUILD_JOBS_VALUE="$(sysctl -n hw.logicalcpu)"
else
    BUILD_JOBS_VALUE=2
fi
[[ "$BUILD_JOBS_VALUE" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: BUILD_JOBS must be a positive integer, got: $BUILD_JOBS_VALUE" >&2
    exit 1
}
make -j"$BUILD_JOBS_VALUE"
popd >/dev/null

mkdir -p AirLib/lib/x64/$folder_name
mkdir -p AirLib/deps/rpclib/lib
mkdir -p AirLib/deps/MavLinkCom/lib
cp $build_dir/output/lib/libAirLib.a AirLib/lib
cp $build_dir/output/lib/libMavLinkCom.a AirLib/deps/MavLinkCom/lib
cp $build_dir/output/lib/librpc.a AirLib/deps/rpclib/lib/librpc.a

# Update AirLib/lib, AirLib/deps, Plugins folders with new binaries
rsync -a --delete $build_dir/output/lib/ AirLib/lib/x64/$folder_name
rsync -a --delete external/rpclib/$RPC_VERSION_FOLDER/include AirLib/deps/rpclib
rsync -a --delete MavLinkCom/include AirLib/deps/MavLinkCom
rsync -a --delete AirLib Unreal/Plugins/AirSim/Source
rm -rf Unreal/Plugins/AirSim/Source/AirLib/src

if [ "$(uname)" == "Darwin" ] && [ "$MAC_ARCH" == "arm64" ] && command -v lipo >/dev/null 2>&1; then
    for artifact in AirLib/lib/libAirLib.a AirLib/deps/MavLinkCom/lib/libMavLinkCom.a AirLib/deps/rpclib/lib/librpc.a; do
        artifact_info="$(lipo -info "$artifact" 2>&1)" || {
            echo "ERROR: could not inspect native artifact: $artifact" >&2
            exit 1
        }
        [[ "$artifact_info" == *arm64* ]] || {
            echo "ERROR: expected arm64 artifact, got: $artifact_info ($artifact)" >&2
            exit 1
        }
    done
    echo "Verified native arm64 AirLib artifacts."
fi

set +x

echo ""
echo ""
echo "==============================="
echo " Cosys-AirSim plugin is built! "
echo "==============================="
echo ""
echo "For further info see for installation see:"
echo "https://github.com/Cosys-Lab/Cosys-AirSim/tree/main/docs/install_linux.md"
echo "=================================================================="

popd >/dev/null
