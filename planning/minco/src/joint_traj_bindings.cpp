#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "optimizer/poly_traj_optimizer.hpp"
#include "traj_utils/poly_traj_utils.hpp"

namespace py = pybind11;

// Degrees of freedom of the robot arm (7-DOF)
constexpr int FRANKA_DOF = 7;
constexpr int POLY_DEGREE = 7;

using FrankaTrajectory = poly_traj::JointTrajectory<POLY_DEGREE, FRANKA_DOF>;
using FrankaOptimizer = joint_traj_opt::JointTrajOptimizer<FRANKA_DOF>;
using FrankaMinSnapOpt = poly_traj::MinSnapOpt<FRANKA_DOF>;

// Helper function to sample trajectory
py::array_t<double> sampleTrajectory(const FrankaTrajectory &traj,
                                     double dt, int derivative_order)
{
    double totalT = traj.getTotalDuration();
    int numSamples = static_cast<int>(std::round(totalT / dt)) + 1;

    py::array_t<double> result({numSamples, FRANKA_DOF});
    auto buf = result.mutable_unchecked<2>();

    for (int i = 0; i < numSamples; ++i)
    {
        double t = std::min(i * dt, totalT - 1e-6);
        Eigen::Matrix<double, FRANKA_DOF, 1> val;

        switch (derivative_order)
        {
        case 0:
            val = traj.getPos(t);
            break;
        case 1:
            val = traj.getVel(t);
            break;
        case 2:
            val = traj.getAcc(t);
            break;
        case 3:
            val = traj.getJer(t);
            break;
        default:
            val = traj.getPos(t);
            break;
        }

        for (int j = 0; j < FRANKA_DOF; ++j)
        {
            buf(i, j) = val(j);
        }
    }

    return result;
}

PYBIND11_MODULE(joint_traj_opt, m)
{
    m.doc() = "Joint Space Trajectory Optimization (MINCO) for robotic arms";

    // JointTrajectory class binding
    py::class_<FrankaTrajectory>(m, "JointTrajectory")
        .def(py::init<>())
        .def("get_piece_num", &FrankaTrajectory::getPieceNum)
        .def("get_dim", &FrankaTrajectory::getDim)
        .def("get_total_duration", &FrankaTrajectory::getTotalDuration)
        .def("get_durations", &FrankaTrajectory::getDurations)
        .def("get_pos", [](const FrankaTrajectory &traj, double t) {
            return traj.getPos(t);
        }, py::arg("t"), "Get position at time t")
        .def("get_vel", [](const FrankaTrajectory &traj, double t) {
            return traj.getVel(t);
        }, py::arg("t"), "Get velocity at time t")
        .def("get_acc", [](const FrankaTrajectory &traj, double t) {
            return traj.getAcc(t);
        }, py::arg("t"), "Get acceleration at time t")
        .def("get_jerk", [](const FrankaTrajectory &traj, double t) {
            return traj.getJer(t);
        }, py::arg("t"), "Get jerk at time t")
        .def("sample", &sampleTrajectory,
             py::arg("dt") = 0.01,
             py::arg("derivative_order") = 0,
             R"doc(
             Sample the trajectory at regular intervals
             
             Args:
                 dt: Time step for sampling
                 derivative_order: 0=position, 1=velocity, 2=acceleration, 3=jerk
             
             Returns:
                 numpy array of shape (num_samples, 7)
             )doc");

    // MinSnapOpt class binding (name kept consistent with the original code)
    py::class_<FrankaMinSnapOpt>(m, "MinSnapOpt")
        .def(py::init<>())
        .def("reset", &FrankaMinSnapOpt::reset, py::arg("piece_num"))
        .def("generate", &FrankaMinSnapOpt::generate,
             py::arg("inner_points"),
             py::arg("times"),
             py::arg("head_state"),
             py::arg("tail_state"))
        .def("get_traj_jerk_cost", &FrankaMinSnapOpt::getTrajJerkCost,
             "Get trajectory cost (actually snap cost, name kept for compatibility)")
        .def("get_traj", &FrankaMinSnapOpt::getTraj)
        .def("get_init_constrain_points", &FrankaMinSnapOpt::getInitConstrainPoints,
             py::arg("K") = 1);

    // JointTrajOptimizer class binding
    py::class_<FrankaOptimizer>(m, "JointTrajOptimizer")
        .def(py::init<>())
        .def("set_param", &FrankaOptimizer::setParam,
             py::arg("min_pos"),
             py::arg("max_pos"),
             py::arg("max_vel"),
             py::arg("max_acc"),
             py::arg("wei_time") = 1.0,
             py::arg("wei_feas") = 1e3,
             py::arg("cps_per_piece") = 5,
             py::arg("fix_waypoints") = false,
             R"doc(
             Set optimization parameters
             
             Args:
                 min_pos: Minimum joint positions (7,)
                 max_pos: Maximum joint positions (7,)
                 max_vel: Maximum joint velocities (7,)
                 max_acc: Maximum joint accelerations (7,)
                 wei_time: Time weight
                 wei_feas: Feasibility weight
                 cps_per_piece: Control points per piece
                 fix_waypoints: If True, trajectory MUST pass through all waypoints (only optimize time)
             )doc")
        .def("set_fix_waypoints", &FrankaOptimizer::setFixWaypoints,
             py::arg("fix"),
             "Set whether to fix intermediate waypoints (if True, only optimize time)")
        .def("set_default_franka_param", &FrankaOptimizer::setDefaultFrankaParam,
             "Set default joint parameters (Franka Emika Panda limits)")
        .def("optimize", [](FrankaOptimizer &opt,
                            const Eigen::MatrixXd &waypoints,
                            const Eigen::VectorXd &initT,
                            const Eigen::MatrixXd &headState,
                            const Eigen::MatrixXd &tailState) {
            Eigen::VectorXd optT;
            FrankaTrajectory optTraj;
            
            bool success = opt.optimizeTrajectory(waypoints, initT, headState, tailState, optT, optTraj);
            
            return py::make_tuple(success, optT, optTraj);
        },
             py::arg("waypoints"),
             py::arg("init_times"),
             py::arg("head_state"),
             py::arg("tail_state"),
             R"doc(
             Optimize trajectory
             
             Args:
                 waypoints: (7, N+2) matrix with start, waypoints, and end
                 init_times: (N+1,) initial time allocation
                 head_state: (7, 4) matrix [pos, vel, acc, jerk] at start
                 tail_state: (7, 4) matrix [pos, vel, acc, jerk] at end
             
             Returns:
                 tuple: (success, optimal_times, trajectory)
             )doc")
        .def("get_iter_num", &FrankaOptimizer::getIterNum,
             "Get number of iterations");

    // Convenience function: quick trajectory optimization
    m.def("optimize_franka_trajectory",
          [](const Eigen::MatrixXd &waypoints,
             const Eigen::VectorXd &initT,
             const Eigen::VectorXd &min_pos,
             const Eigen::VectorXd &max_pos,
             const Eigen::VectorXd &max_vel,
             const Eigen::VectorXd &max_acc,
             double wei_time,
             double wei_feas,
             bool fix_waypoints) {
              FrankaOptimizer opt;
              opt.setParam(min_pos, max_pos, max_vel, max_acc, wei_time, wei_feas, 5, fix_waypoints);

              Eigen::MatrixXd headState = Eigen::MatrixXd::Zero(FRANKA_DOF, 4);
              Eigen::MatrixXd tailState = Eigen::MatrixXd::Zero(FRANKA_DOF, 4);
              headState.col(0) = waypoints.col(0);
              tailState.col(0) = waypoints.col(waypoints.cols() - 1);

              Eigen::VectorXd optT;
              FrankaTrajectory optTraj;

              bool success = opt.optimizeTrajectory(waypoints, initT, headState, tailState, optT, optTraj);

              return py::make_tuple(success, optT, optTraj);
          },
          py::arg("waypoints"),
          py::arg("init_times"),
          py::arg("min_pos"),
          py::arg("max_pos"),
          py::arg("max_vel"),
          py::arg("max_acc"),
          py::arg("wei_time") = 1.0,
          py::arg("wei_feas") = 1e3,
          py::arg("fix_waypoints") = false,
          R"doc(
          Quick optimization function for a joint-space trajectory
          
          Args:
              waypoints: (7, N+2) waypoints matrix
              init_times: (N+1,) initial time allocation
              min_pos: (7,) minimum joint positions
              max_pos: (7,) maximum joint positions
              max_vel: (7,) maximum joint velocities
              max_acc: (7,) maximum joint accelerations
              wei_time: Time weight
              wei_feas: Feasibility weight
              fix_waypoints: If True, trajectory MUST pass through all waypoints
          
          Returns:
              tuple: (success, optimal_times, trajectory)
          )doc");

    // Quick optimization using the default joint parameters
    m.def("optimize_franka_trajectory_default",
          [](const Eigen::MatrixXd &waypoints,
             const Eigen::VectorXd &initT,
             bool fix_waypoints) {
              FrankaOptimizer opt;
              opt.setDefaultFrankaParam();
              opt.setFixWaypoints(fix_waypoints);

              Eigen::MatrixXd headState = Eigen::MatrixXd::Zero(FRANKA_DOF, 4);
              Eigen::MatrixXd tailState = Eigen::MatrixXd::Zero(FRANKA_DOF, 4);
              headState.col(0) = waypoints.col(0);
              tailState.col(0) = waypoints.col(waypoints.cols() - 1);

              Eigen::VectorXd optT;
              FrankaTrajectory optTraj;

              bool success = opt.optimizeTrajectory(waypoints, initT, headState, tailState, optT, optTraj);

              return py::make_tuple(success, optT, optTraj);
          },
          py::arg("waypoints"),
          py::arg("init_times"),
          py::arg("fix_waypoints") = false,
          R"doc(
          Quick optimization with default joint parameters (Franka Emika Panda limits)
          
          Args:
              waypoints: (7, N+2) waypoints matrix
              init_times: (N+1,) initial time allocation
              fix_waypoints: If True, trajectory MUST pass through all waypoints
          
          Returns:
              tuple: (success, optimal_times, trajectory)
          )doc");

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("DOF") = FRANKA_DOF;
    m.attr("POLY_DEGREE") = POLY_DEGREE;
}
