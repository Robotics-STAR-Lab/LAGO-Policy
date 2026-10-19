/**
 * @file pybind_ego_planner.cpp
 * @brief Pybind11 bindings for ego-planner B-spline trajectory optimization
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "uniform_bspline.hpp"
#include "bspline_optimizer.hpp"
#include "polynomial_traj.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ego_planner_cpp, m) {
    m.doc() = "Ego-planner B-spline trajectory optimization for robotic arm obstacle avoidance";
    
    // ==================== AABB Obstacle ====================
    py::class_<ego_planner::AABB>(m, "AABB")
        .def(py::init<>())
        .def(py::init<const Eigen::Vector3d&, const Eigen::Vector3d&>(),
             py::arg("min_corner"), py::arg("max_corner"))
        .def_readwrite("min_corner", &ego_planner::AABB::min_corner)
        .def_readwrite("max_corner", &ego_planner::AABB::max_corner)
        .def("center", [](const ego_planner::AABB& self) {
            return ego_planner::aabbCenter(self);
        })
        .def("half_size", [](const ego_planner::AABB& self) {
            return ego_planner::aabbHalfSize(self);
        });
    
    // ==================== UniformBspline ====================
    py::class_<ego_planner::UniformBspline>(m, "UniformBspline")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&, int, double>(),
             py::arg("control_points"), py::arg("order"), py::arg("interval"),
             "Create B-spline from control points (3 x N matrix)")
        .def("get_control_points", &ego_planner::UniformBspline::getControlPoints)
        .def("get_interval", &ego_planner::UniformBspline::getInterval)
        .def("get_order", &ego_planner::UniformBspline::getOrder)
        .def("get_duration", &ego_planner::UniformBspline::getDuration)
        .def("get_length", &ego_planner::UniformBspline::getLength,
             py::arg("resolution") = 0.01)
        .def("evaluate", &ego_planner::UniformBspline::evaluateDeBoorT,
             py::arg("t"),
             "Evaluate B-spline at time t (t=0 is start)")
        .def("sample_trajectory", &ego_planner::UniformBspline::sampleTrajectory,
             py::arg("dt"),
             "Sample trajectory at uniform time intervals, returns 3 x N matrix")
        .def("get_derivative", &ego_planner::UniformBspline::getDerivative,
             "Get derivative B-spline (velocity curve)")
        .def("set_physical_limits", &ego_planner::UniformBspline::setPhysicalLimits,
             py::arg("max_vel"), py::arg("max_acc"), py::arg("tolerance") = 0.0)
        .def("check_feasibility", [](ego_planner::UniformBspline& self) {
            double ratio;
            bool feasible = self.checkFeasibility(ratio);
            return std::make_pair(feasible, ratio);
        }, "Check if trajectory satisfies velocity/acceleration limits")
        .def("lengthen_time", &ego_planner::UniformBspline::lengthenTime,
             py::arg("ratio"),
             "Lengthen time to satisfy constraints");
    
    // ==================== Helper Functions ====================
    
    // parameterizeToBspline: Generate B-spline control points with velocity/acceleration constraints
    // This is the key function from original ego-planner for smooth replanning
    m.def("parameterize_to_bspline", [](
        double ts,
        const std::vector<Eigen::Vector3d>& point_set,
        const Eigen::Vector3d& start_vel,
        const Eigen::Vector3d& end_vel,
        const Eigen::Vector3d& start_acc,
        const Eigen::Vector3d& end_acc
    ) -> Eigen::MatrixXd {
        return ego_planner::parameterizeToBsplineSimple(ts, point_set, start_vel, end_vel, start_acc, end_acc);
    },
    py::arg("ts"),
    py::arg("point_set"),
    py::arg("start_vel"),
    py::arg("end_vel"),
    py::arg("start_acc") = Eigen::Vector3d::Zero(),
    py::arg("end_acc") = Eigen::Vector3d::Zero(),
    R"doc(
    Generate B-spline control points from path points with velocity/acceleration constraints.
    
    This function is essential for smooth replanning, as it ensures the generated trajectory
    satisfies the specified velocity and acceleration at start and end points.
    
    Parameters:
    - ts: Time interval between control points (seconds)
    - point_set: List of K path points that the B-spline should pass through
    - start_vel: Velocity at the start (for continuity with previous trajectory)
    - end_vel: Velocity at the end (typically zero for goal)
    - start_acc: Acceleration at start (default: zero)
    - end_acc: Acceleration at end (default: zero)
    
    Returns:
    - Control points matrix (3 x (K+2))
    )doc");
    
    // ==================== BsplineOptimizer ====================
    py::class_<ego_planner::BsplineOptimizer>(m, "BsplineOptimizer")
        .def(py::init<>())
        
        // Parameter settings
        .def("set_weights", &ego_planner::BsplineOptimizer::setWeights,
             py::arg("smooth") = 1.0,
             py::arg("collision") = 10.0,
             py::arg("feasibility") = 0.1,
             py::arg("fitness") = 1.0,
             R"doc(
             Set optimization weights.
             
             Stage 1 (Rebound): uses smooth, collision, feasibility
             Stage 2 (Refine): uses smooth, feasibility, fitness
             
             Parameters:
             - smooth: Jerk smoothness weight (lambda1)
             - collision: Collision avoidance weight (lambda2)
             - feasibility: Velocity/acceleration constraint weight (lambda3)
             - fitness: Curve fitting weight for Stage 2 (lambda4)
             )doc")
        .def("set_physical_limits", &ego_planner::BsplineOptimizer::setPhysicalLimits,
             py::arg("max_vel") = 0.5,
             py::arg("max_acc") = 2.0,
             py::arg("safe_dist") = 0.05,
             R"doc(
             Set physical constraints.
             
             Parameters:
             - max_vel: Maximum velocity (m/s) for each axis
             - max_acc: Maximum acceleration (m/s^2) for each axis
             - safe_dist: Safety clearance from obstacles (m)
             )doc")
        .def("set_order", &ego_planner::BsplineOptimizer::setOrder,
             py::arg("order") = 3,
             "Set B-spline order/degree (typically 3 for cubic)")
        .def("get_order", &ego_planner::BsplineOptimizer::getOrder)
        .def("get_max_vel", &ego_planner::BsplineOptimizer::getMaxVel)
        .def("get_max_acc", &ego_planner::BsplineOptimizer::getMaxAcc)
        .def("get_safe_dist", &ego_planner::BsplineOptimizer::getSafeDist)
        
        // Obstacle management
        .def("clear_obstacles", &ego_planner::BsplineOptimizer::clearObstacles)
        .def("add_obstacle", 
             py::overload_cast<const ego_planner::AABB&>(&ego_planner::BsplineOptimizer::addObstacle),
             py::arg("obstacle"))
        .def("add_obstacle",
             py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&>(&ego_planner::BsplineOptimizer::addObstacle),
             py::arg("min_corner"), py::arg("max_corner"))
        .def("set_obstacles", &ego_planner::BsplineOptimizer::setObstacles,
             py::arg("obstacles"))
        .def("get_obstacles", &ego_planner::BsplineOptimizer::getObstacles)
        
        // Optimization
        .def("optimize", 
             py::overload_cast<const Eigen::Vector3d&, const Eigen::Vector3d&, int, double, int>(
                 &ego_planner::BsplineOptimizer::optimizeTrajectory),
             py::arg("start"),
             py::arg("goal"),
             py::arg("num_ctrl_pts") = 15,
             py::arg("ts") = 0.1,
             py::arg("max_iterations") = 100,
             R"doc(
             Optimize trajectory from start to goal.
             
             Parameters:
             - start: Start position (3D vector)
             - goal: Goal position (3D vector)
             - num_ctrl_pts: Number of control points. 
                 Paper suggests ~25 for 7m horizon with 0.3m interval.
                 For robotic arm (~0.5m workspace), use 10-15.
             - ts: Time interval between control points (seconds).
                 Smaller = smoother but slower trajectory.
             - max_iterations: Maximum optimization iterations.
             
             Returns: Optimized UniformBspline trajectory
             )doc")
        .def("optimize_from_ctrl_pts",
             py::overload_cast<const Eigen::MatrixXd&, double, int>(
                 &ego_planner::BsplineOptimizer::optimizeTrajectory),
             py::arg("init_ctrl_pts"),
             py::arg("ts") = 0.1,
             py::arg("max_iterations") = 100,
             "Optimize from given initial control points (3 x N matrix)")
        
        // Collision checking
        .def("is_collision", &ego_planner::BsplineOptimizer::isCollision,
             py::arg("point"),
             "Check if point is in collision with any obstacle (O(1) with GridMap)")
        .def("get_min_distance", &ego_planner::BsplineOptimizer::getMinDistance,
             py::arg("point"),
             "Get minimum signed distance to obstacles (negative = inside)")
        .def("get_control_points", &ego_planner::BsplineOptimizer::getControlPoints)
        
        // GridMap control
        .def("set_use_grid_map", &ego_planner::BsplineOptimizer::setUseGridMap,
             py::arg("use") = true,
             R"doc(
             Enable/disable GridMap for O(1) collision detection.
             
             When enabled (default), a 3D occupancy grid is pre-computed
             from obstacles at the start of optimization. Collision checks
             become O(1) table lookups instead of O(n) AABB traversals.
             
             This is similar to ego-planner's GridMap approach.
             )doc")
        .def("get_use_grid_map", &ego_planner::BsplineOptimizer::getUseGridMap,
             "Check if GridMap is enabled")
        .def("init_grid_map", &ego_planner::BsplineOptimizer::initGridMap,
             R"doc(
             Manually initialize GridMap after adding obstacles.
             
             This is called automatically during optimize(), but you can
             call it manually if you want to use is_collision() before
             optimization.
             )doc")
        
        // Ground level constraint for robotic arm
        .def("set_ground_level", &ego_planner::BsplineOptimizer::setGroundLevel,
             py::arg("ground_z"),
             R"doc(
             Set ground/table level constraint for A* search.
             
             A* will not search below this Z level, forcing upward avoidance
             for robotic arm scenarios. This prevents the path from going
             through the table.
             
             Parameters:
             - ground_z: Minimum Z level (meters). Default is 0.05m.
             )doc")
        .def("get_ground_level", &ego_planner::BsplineOptimizer::getGroundLevel,
             "Get current ground level constraint")
        
        // Vertical preference for A* search
        .def("set_prefer_vertical", &ego_planner::BsplineOptimizer::setPreferVertical,
             py::arg("prefer") = true,
             py::arg("discount") = 0.3,
             R"doc(
             Enable preference for vertical (Z-axis) avoidance paths.
             
             When enabled, A* will strongly prefer paths that go UP rather than
             sideways. This is the recommended setting for robotic arm obstacle
             avoidance.
             
             Parameters:
             - prefer: Enable vertical preference (default: True)
             - discount: Cost discount for Z movement. Lower = stronger preference.
                        0.3 means Z movement costs 30% of horizontal movement.
                        0.5 means Z movement costs 50% of horizontal movement.
             )doc")
        .def("get_prefer_vertical", &ego_planner::BsplineOptimizer::getPreferVertical)
        
        // Verbose debug output control
        .def("set_verbose", &ego_planner::BsplineOptimizer::setVerbose,
             py::arg("verbose") = false,
             "Enable/disable verbose debug output (default: off for performance)")
        .def("get_verbose", &ego_planner::BsplineOptimizer::getVerbose,
             "Check if vertical preference is enabled")
        
        // Fixed workspace for faster repeated optimizations
        .def("set_workspace", &ego_planner::BsplineOptimizer::setWorkspace,
             py::arg("min_pt"),
             py::arg("max_pt"),
             R"doc(
             Set fixed workspace bounds for faster repeated optimizations.
             
             When set, the optimizer pre-initializes GridMap once and reuses it
             for all subsequent optimizations. This avoids the ~2ms overhead of
             GridMap initialization on each call.
             
             Parameters:
             - min_pt: Minimum corner of workspace (x, y, z) as numpy array
             - max_pt: Maximum corner of workspace (x, y, z) as numpy array
             
             Example:
                 optimizer.set_workspace([0.1, -0.5, 0.05], [0.8, 0.5, 0.5])
             )doc")
        .def("clear_workspace", &ego_planner::BsplineOptimizer::clearWorkspace,
             "Clear fixed workspace and use auto-computed bounds")
        .def("has_fixed_workspace", &ego_planner::BsplineOptimizer::hasFixedWorkspace,
             "Check if fixed workspace is set");
    
    // ==================== PolynomialTraj ====================
    // This is essential for smooth replanning with velocity/acceleration continuity
    py::class_<ego_planner::PolynomialTraj>(m, "PolynomialTraj")
        .def(py::init<>())
        .def("evaluate", &ego_planner::PolynomialTraj::evaluate,
             py::arg("t"),
             "Evaluate position at time t")
        .def("evaluate_vel", &ego_planner::PolynomialTraj::evaluateVel,
             py::arg("t"),
             "Evaluate velocity at time t")
        .def("evaluate_acc", &ego_planner::PolynomialTraj::evaluateAcc,
             py::arg("t"),
             "Evaluate acceleration at time t")
        .def("get_total_time", &ego_planner::PolynomialTraj::getTotalTime,
             "Get total duration of trajectory")
        .def("sample_trajectory", &ego_planner::PolynomialTraj::sampleTrajectory,
             py::arg("dt"),
             "Sample trajectory at regular intervals")
        .def_static("one_segment_traj_gen", &ego_planner::PolynomialTraj::one_segment_traj_gen,
             py::arg("start_pt"), py::arg("start_vel"), py::arg("start_acc"),
             py::arg("end_pt"), py::arg("end_vel"), py::arg("end_acc"),
             py::arg("duration"),
             R"doc(
             Generate a single-segment polynomial trajectory satisfying boundary conditions.
             
             This is THE KEY function from ego-planner for smooth replanning.
             It generates a 5th-order polynomial that EXACTLY satisfies:
             - Start position, velocity, acceleration
             - End position, velocity, acceleration
             
             Usage for replanning:
             1. Get current pos/vel/acc from current trajectory
             2. Generate polynomial traj to goal with these as start conditions
             3. Sample points from polynomial trajectory
             4. Use parameterize_to_bspline to get initial control points
             5. Optimize with optimize_from_ctrl_pts
             
             Parameters:
             - start_pt: Start position
             - start_vel: Start velocity (from current trajectory)
             - start_acc: Start acceleration (from current trajectory)
             - end_pt: End position (goal)
             - end_vel: End velocity (typically zero)
             - end_acc: End acceleration (typically zero)
             - duration: Duration of the trajectory
             
             Returns:
             - PolynomialTraj with one segment
             )doc");
    
    // ==================== Utility Functions ====================
    m.def("straight_line_control_points", &ego_planner::straightLineControlPoints,
          py::arg("start"), py::arg("end"), py::arg("num_ctrl_pts"),
          "Generate straight-line control points from start to end");
    
    m.def("signed_distance_to_aabb", &ego_planner::signedDistanceToAABB,
          py::arg("point"), py::arg("box"),
          "Compute signed distance from point to AABB (negative = inside)");
}

