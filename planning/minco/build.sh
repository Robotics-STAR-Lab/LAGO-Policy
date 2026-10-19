#!/bin/bash
# Build script for the MINCO joint-space trajectory optimizer
# (joint_traj_opt pybind11 module).
#
# Activate your environment first (e.g. `conda activate lago_policy`) so that the
# build links against the right Python and finds pybind11 (`pip install pybind11`).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build into ./build (the default MINCO_BUILD_DIR expected by the wrapper)
mkdir -p build
cd build

# Configure and build, pinning to the active Python interpreter
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(which python)"
make -j$(nproc)

echo ""
echo "Build successful!"
echo "Module built at: ${SCRIPT_DIR}/build/joint_traj_opt.*.so"
