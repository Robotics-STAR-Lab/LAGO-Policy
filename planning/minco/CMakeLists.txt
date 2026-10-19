cmake_minimum_required(VERSION 3.14)
project(joint_traj_opt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find required packages
find_package(Eigen3 REQUIRED)

# Use specified Python executable if provided, otherwise find system Python
if(DEFINED PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Use pybind11 - prefer command line pybind11_DIR, otherwise auto-detect the
# pip-installed pybind11 in the active Python environment (`pip install pybind11`).
if(NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_cmakedir_result
    )
endif()
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/traj_opt/include
    ${CMAKE_CURRENT_SOURCE_DIR}/traj_utils/include
    ${pybind11_INCLUDE_DIRS}
)

# Create pybind11 module
pybind11_add_module(joint_traj_opt
    src/joint_traj_bindings.cpp
)

target_include_directories(joint_traj_opt PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/traj_opt/include
    ${CMAKE_CURRENT_SOURCE_DIR}/traj_utils/include
)

target_compile_options(joint_traj_opt PRIVATE
    -O3
    -march=native
    -ffast-math
)

# Install target
install(TARGETS joint_traj_opt
    LIBRARY DESTINATION ${Python3_SITELIB}
)

# Print configuration info
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Eigen3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "pybind11_VERSION: ${pybind11_VERSION}")
