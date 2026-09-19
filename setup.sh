#! /bin/bash
set -x
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
pushd "$SCRIPT_DIR" >/dev/null

downloadHighPolySuv=true
MIN_CMAKE_VERSION=3.12.0

# Install a Homebrew formula only when it is not already present. Formulae
# such as llvm@18 are keg-only, so checking the formula database is more
# reliable than relying on PATH links.
function brew_install() { brew list --formula "$1" &>/dev/null || brew install "$1"; }

# BSD sort on macOS does not implement GNU sort's -V option. Homebrew's
# coreutils provides gsort; retain the GNU sort path on Linux and use a small
# numeric fallback if neither implementation is available.
function version_less_than_equal_to() {
    local first="$1"
    local second="$2"
    local sort_command=sort
    if command -v gsort >/dev/null 2>&1; then
        sort_command=gsort
    fi
    if "$sort_command" -V </dev/null >/dev/null 2>&1; then
        test "$(printf '%s\n' "$first" "$second" | "$sort_command" -V | head -n 1)" = "$first"
        return
    fi
    awk -v first="$first" -v second="$second" '
        function normalized(value, parts, count, result, i) {
            count = split(value, parts, ".")
            result = ""
            for (i = 1; i <= 3; i++) result = result sprintf("%03d", (i <= count ? parts[i] + 0 : 0))
            return result
        }
        BEGIN { exit !(normalized(first) <= normalized(second)) }
    '
}

# Parse command line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --no-full-poly-car)
    downloadHighPolySuv=false
    shift # past value
    ;;
esac
done

# Homebrew metadata updates are opt-in so setup does not trigger a blanket
# refresh on an already configured Mac. Set HERCULES_BREW_UPDATE=1 when needed.
# llvm tools
if [ "$(uname)" == "Darwin" ]; then # osx
    command -v brew >/dev/null 2>&1 || {
        echo "ERROR: Homebrew is required on macOS. Install it from https://brew.sh/" >&2
        exit 1
    }
    if [[ "${HERCULES_BREW_UPDATE:-0}" == "1" ]]; then brew update; fi
    # Update below line for newer versions
    brew_install llvm@18
else # linux
    sudo apt-get update
    sudo apt-get -y install --no-install-recommends \
        lsb-release \
        rsync \
        software-properties-common \
        wget \
        libvulkan1 \
        vulkan-tools

    # install clang and build tools
    VERSION=$(lsb_release -rs | cut -d. -f1)
    if [ "$VERSION" -ge "20" ]; then
        clang_version='12'
        cpp_version='12'
    else
        clang_version='12'
        cpp_version='10'
    fi
    sudo apt-get install -y \
        clang-$clang_version \
        clang++-$clang_version \
        libc++-$clang_version-dev \
        libc++abi-$clang_version-dev \
        libstdc++-$cpp_version-dev
fi

if ! which cmake; then
    # CMake not installed
    cmake_ver=0
else
    cmake_ver=$(cmake --version 2>&1 | head -n1 | cut -d ' ' -f3 | awk '{print $NF}')
fi

#give user perms to access USB port - this is not needed if not using PX4 HIL
#TODO: figure out how to do below in travis
# Install additional tools, CMake if required
if [ "$(uname)" == "Darwin" ]; then # osx
    if command -v dseditgroup >/dev/null 2>&1 && [[ -n "${USER:-}" ]]; then #this happens when running in travis
        sudo dseditgroup -o edit -a "$USER" -t user dialout
    fi

    # MacOS 11 has new Python env management that breaks the Python 2-to-3
    # build process. We need to make sure brew updates before attempting to
    # install, since it will update packaages
    if [[ "${HERCULES_BREW_UPDATE:-0}" == "1" ]]; then brew update; fi
    brew_install wget
    brew_install coreutils

    if version_less_than_equal_to $cmake_ver $MIN_CMAKE_VERSION; then
        brew_install cmake
    else
        echo "Already have good version of cmake: $cmake_ver"
    fi

else #linux
    if [[ ! -z "${whoami}" ]]; then #this happens when running in travis
        sudo /usr/sbin/useradd -G dialout $USER
        sudo usermod -a -G dialout $USER
    fi

    # install additional tools
    sudo apt-get install -y build-essential unzip libunwind-dev

    if version_less_than_equal_to $cmake_ver $MIN_CMAKE_VERSION; then
        VERSION=$(lsb_release -rs | cut -d. -f1)
        # For Ubuntu 18 and up, avoid building cmake from scratch to save time
        # ref: https://apt.kitware.com
        if [ "$VERSION" -ge "18" ]; then
            sudo apt-get -y install \
                apt-transport-https \
                ca-certificates \
                gnupg
            wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
            sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
            sudo apt-get -y install --no-install-recommends \
                make \
                cmake

        else
            # For Ubuntu 16.04, or anything else, build CMake 3.10.2 from source
            if [[ ! -d "cmake_build/bin" ]]; then
                echo "Downloading cmake..."
                wget https://cmake.org/files/v3.10/cmake-3.10.2.tar.gz \
                    -O cmake.tar.gz
                tar -xzf cmake.tar.gz
                rm cmake.tar.gz
                rm -rf ./cmake_build
                mv ./cmake-3.10.2 ./cmake_build
                pushd cmake_build
                ./bootstrap
                make
                popd
            fi
        fi

    else
        echo "Already have good version of cmake: $cmake_ver"
    fi

fi # End USB setup, CMake install


# Download rpclib
if [ ! -d "external/rpclib/rpclib-2.3.0" ]; then
    echo "*********************************************************************************************"
    echo "Downloading rpclib..."
    echo "*********************************************************************************************"

    wget https://github.com/rpclib/rpclib/archive/v2.3.0.zip

    # remove previous versions
    rm -rf "external/rpclib"

    mkdir -p "external/rpclib"
    unzip -q v2.3.0.zip -d external/rpclib
    rm v2.3.0.zip
fi

# Download high-polycount SUV model
if $downloadHighPolySuv; then
    if [ ! -d "Unreal/Plugins/AirSim/Content/VehicleAdv" ]; then
        mkdir -p "Unreal/Plugins/AirSim/Content/VehicleAdv"
    fi
    if [ ! -d "Unreal/Plugins/AirSim/Content/VehicleAdv/SUV/v1.2.0" ]; then
            echo "*********************************************************************************************"
            echo "Downloading high-poly car assets.... The download is ~37MB and can take some time."
            echo "To install without this assets, re-run setup.sh with the argument --no-full-poly-car"
            echo "*********************************************************************************************"

            if [ -d "suv_download_tmp" ]; then
                rm -rf "suv_download_tmp"
            fi
            mkdir -p "suv_download_tmp"
            cd suv_download_tmp
            wget  https://github.com/Cosys-Lab/Cosys-AirSim/releases/download/carassets/cosys_car_assets.zip
            if [ -d "../Unreal/Plugins/AirSim/Content/VehicleAdv/SUV" ]; then
                rm -rf "../Unreal/Plugins/AirSim/Content/VehicleAdv/SUV"
            fi
            unzip -q cosys_car_assets.zip -d ../Unreal/Plugins/AirSim/Content/VehicleAdv
            cd ..
            rm -rf "suv_download_tmp"
    fi
else
    echo "### Not downloading high-poly car asset (--no-full-poly-car). The default unreal vehicle will be used."
fi

echo "Installing Eigen library..."

if [ ! -d "AirLib/deps/eigen3" ]; then
    echo "Downloading Eigen..."
    wget -O eigen3.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
    unzip -q eigen3.zip -d temp_eigen
    mkdir -p AirLib/deps/eigen3
    mv temp_eigen/eigen*/Eigen AirLib/deps/eigen3
    rm -rf temp_eigen
    rm eigen3.zip
else
    echo "Eigen is already installed."
fi

popd >/dev/null

set +x
echo ""
echo ""
echo "============================================"
echo " Cosys-AirSim setup completed successfully! "
echo "============================================"
echo ""
echo "Run ./build.sh to compile."
