#!/bin/bash
# Build script for ego_planner_cpp (the ego-planner pybind11 module)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create build directory
mkdir -p build
cd build

# Configure and build, pinning to the active Python interpreter (so pybind11 is
# found via `python -m pybind11 --cmakedir` and the .so matches this Python).
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(which python)"
make -j$(nproc)

echo ""
echo "Build successful!"
echo "Module built at: ${SCRIPT_DIR}/build/ego_planner_cpp*.so"




