cmake_minimum_required(VERSION 3.14)
project(ego_planner_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Release with debug info by default for good performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find packages
find_package(Eigen3 REQUIRED)

# Use the specified Python interpreter if provided, otherwise find system Python.
if(DEFINED PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11: prefer a command-line pybind11_DIR, otherwise auto-detect the
# pip-installed pybind11 in the active Python env (`pip install pybind11`).
if(NOT DEFINED pybind11_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Create pybind11 module
pybind11_add_module(ego_planner_cpp
    src/pybind_ego_planner.cpp
)

target_link_libraries(ego_planner_cpp PRIVATE
    Eigen3::Eigen
)

# Emit the .so into ./build (where planning/ego_planner_manager.py looks for it)
set_target_properties(ego_planner_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)

# Print configuration info
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")




