/**
 * @file uniform_bspline.hpp
 * @brief Uniform B-spline implementation for trajectory representation
 * 
 * Based on ego-planner's uniform_bspline but standalone (no ROS dependency)
 */

#ifndef UNIFORM_BSPLINE_HPP
#define UNIFORM_BSPLINE_HPP

#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <cmath>

namespace ego_planner {

class UniformBspline {
public:
    UniformBspline() : p_(3), n_(0), m_(0), interval_(0.1) {}
    
    /**
     * @brief Construct B-spline from control points
     * @param points Control points matrix (3 x N)
     * @param order B-spline order (degree), typically 3
     * @param interval Knot span (time interval between control points)
     */
    UniformBspline(const Eigen::MatrixXd& points, int order, double interval)
        : p_(order), interval_(interval) {
        setUniformBspline(points, order, interval);
    }
    
    ~UniformBspline() = default;
    
    /**
     * @brief Initialize as uniform B-spline (same as original ego-planner)
     */
    void setUniformBspline(const Eigen::MatrixXd& points, int order, double interval) {
        control_points_ = points;
        p_ = order;
        interval_ = interval;
        n_ = points.cols() - 1;
        m_ = n_ + p_ + 1;
        
        // Construct uniform knot vector (EXACTLY like original ego-planner)
        // Original uses: u_(i) = (-p_ + i) * interval_ for i <= p_
        // This makes u_(p_) = 0, which is the start of the valid parameter range
        u_ = Eigen::VectorXd::Zero(m_ + 1);
        for (int i = 0; i <= m_; ++i) {
            if (i <= p_) {
                u_(i) = double(-p_ + i) * interval_;
            } else if (i > p_ && i <= m_ - p_) {
                u_(i) = u_(i - 1) + interval_;
            } else {  // i > m_ - p_
                u_(i) = u_(i - 1) + interval_;
            }
        }
    }
    
    // Getters
    Eigen::MatrixXd getControlPoints() const { return control_points_; }
    double getInterval() const { return interval_; }
    int getOrder() const { return p_; }
    Eigen::VectorXd getKnot() const { return u_; }
    
    /**
     * @brief Get valid time span [u_min, u_max] for evaluation
     */
    bool getTimeSpan(double& u_min, double& u_max) const {
        if (n_ < 0) return false;
        u_min = u_(p_);
        u_max = u_(m_ - p_);
        return true;
    }
    
    /**
     * @brief Evaluate B-spline at parameter u using De Boor's algorithm
     * @param u Parameter value (must be in [u_p, u_{m-p}])
     * @return Point on the curve at parameter u
     * 
     * EXACTLY matches original ego-planner implementation
     */
    Eigen::Vector3d evaluateDeBoor(double u) const {
        // Clamp to valid range (same as original)
        double ub = std::min(std::max(u_(p_), u), u_(m_ - p_));
        
        // Determine which [u_i, u_{i+1}] interval contains ub
        int k = p_;
        while (true) {
            if (u_(k + 1) >= ub)
                break;
            ++k;
        }
        
        // De Boor's algorithm (same as original)
        std::vector<Eigen::Vector3d> d;
        for (int i = 0; i <= p_; ++i) {
            d.push_back(control_points_.col(k - p_ + i));
        }
        
        for (int r = 1; r <= p_; ++r) {
            for (int i = p_; i >= r; --i) {
                double alpha = (ub - u_(i + k - p_)) / (u_(i + 1 + k - r) - u_(i + k - p_));
                d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i];
            }
        }
        
        return d[p_];
    }
    
    /**
     * @brief Evaluate at time t (convenience function: t starts from 0)
     */
    Eigen::Vector3d evaluateDeBoorT(double t) const {
        return evaluateDeBoor(t + u_(p_));
    }
    
    /**
     * @brief Get total duration of the B-spline curve
     */
    double getDuration() const {
        double u_min, u_max;
        if (!getTimeSpan(u_min, u_max)) return 0.0;
        return u_max - u_min;
    }
    
    /**
     * @brief Get arc length of the B-spline curve
     */
    double getLength(double resolution = 0.01) const {
        double u_min, u_max;
        if (!getTimeSpan(u_min, u_max)) return 0.0;
        
        double length = 0.0;
        Eigen::Vector3d prev = evaluateDeBoor(u_min);
        
        for (double u = u_min + resolution; u <= u_max; u += resolution) {
            Eigen::Vector3d curr = evaluateDeBoor(u);
            length += (curr - prev).norm();
            prev = curr;
        }
        
        return length;
    }
    
    /**
     * @brief Sample the trajectory at uniform time intervals
     * @param dt Time step
     * @return Sampled points (3 x N)
     */
    Eigen::MatrixXd sampleTrajectory(double dt) const {
        double u_min, u_max;
        if (!getTimeSpan(u_min, u_max)) {
            return Eigen::MatrixXd();
        }
        
        int num_samples = static_cast<int>((u_max - u_min) / dt) + 1;
        Eigen::MatrixXd samples(3, num_samples);
        
        for (int i = 0; i < num_samples; ++i) {
            double u = u_min + i * dt;
            u = std::min(u, u_max);
            samples.col(i) = evaluateDeBoor(u);
        }
        
        return samples;
    }
    
    /**
     * @brief Get derivative control points (same as original ego-planner)
     * The derivative of a b-spline is also a b-spline, its order becomes p_-1
     * control point Qi = p_*(Pi+1-Pi)/(ui+p_+1-ui+1)
     */
    Eigen::MatrixXd getDerivativeControlPoints() const {
        Eigen::MatrixXd ctp(control_points_.rows(), control_points_.cols() - 1);
        for (int i = 0; i < ctp.cols(); ++i) {
            ctp.col(i) = p_ * (control_points_.col(i + 1) - control_points_.col(i)) 
                         / (u_(i + p_ + 1) - u_(i + 1));
        }
        return ctp;
    }
    
    /**
     * @brief Get derivative B-spline (same as original ego-planner)
     */
    UniformBspline getDerivative() const {
        Eigen::MatrixXd ctp = getDerivativeControlPoints();
        UniformBspline derivative(ctp, p_ - 1, interval_);
        
        // Cut the first and last knot (same as original)
        Eigen::VectorXd knot(u_.rows() - 2);
        knot = u_.segment(1, u_.rows() - 2);
        derivative.setKnot(knot);
        
        return derivative;
    }
    
    /**
     * @brief Set knot vector
     */
    void setKnot(const Eigen::VectorXd& knot) {
        u_ = knot;
    }
    
    /**
     * @brief Set physical limits for feasibility check (same as original)
     */
    void setPhysicalLimits(double max_vel, double max_acc, double tolerance = 0.0) {
        limit_vel_ = max_vel;
        limit_acc_ = max_acc;
        feasibility_tolerance_ = tolerance;
    }
    
    /**
     * @brief Check if trajectory satisfies velocity and acceleration constraints
     * @param ratio Output: ratio to lengthen time if infeasible
     * @return true if feasible
     * 
     * EXACTLY matches original ego-planner implementation
     */
    bool checkFeasibility(double& ratio) const {
        bool fea = true;
        const Eigen::MatrixXd& P = control_points_;
        int dimension = control_points_.rows();
        
        // Check velocity feasibility (same as original)
        double max_vel = -1.0;
        double enlarged_vel_lim = limit_vel_ * (1.0 + feasibility_tolerance_) + 1e-4;
        for (int i = 0; i < static_cast<int>(P.cols()) - 1; ++i) {
            Eigen::Vector3d vel = p_ * (P.col(i + 1) - P.col(i)) / (u_(i + p_ + 1) - u_(i + 1));
            
            if (std::abs(vel(0)) > enlarged_vel_lim || 
                std::abs(vel(1)) > enlarged_vel_lim ||
                std::abs(vel(2)) > enlarged_vel_lim) {
                fea = false;
                for (int j = 0; j < dimension; ++j) {
                    max_vel = std::max(max_vel, std::abs(vel(j)));
                }
            }
        }
        
        // Check acceleration feasibility (same as original)
        double max_acc = -1.0;
        double enlarged_acc_lim = limit_acc_ * (1.0 + feasibility_tolerance_) + 1e-4;
        for (int i = 0; i < static_cast<int>(P.cols()) - 2; ++i) {
            Eigen::Vector3d acc = p_ * (p_ - 1) *
                ((P.col(i + 2) - P.col(i + 1)) / (u_(i + p_ + 2) - u_(i + 2)) -
                 (P.col(i + 1) - P.col(i)) / (u_(i + p_ + 1) - u_(i + 1))) /
                (u_(i + p_ + 1) - u_(i + 2));
            
            if (std::abs(acc(0)) > enlarged_acc_lim || 
                std::abs(acc(1)) > enlarged_acc_lim ||
                std::abs(acc(2)) > enlarged_acc_lim) {
                fea = false;
                for (int j = 0; j < dimension; ++j) {
                    max_acc = std::max(max_acc, std::abs(acc(j)));
                }
            }
        }
        
        ratio = std::max(max_vel / limit_vel_, std::sqrt(std::abs(max_acc) / limit_acc_));
        if (ratio < 0) ratio = 1.0;  // Handle case when no violations found
        
        return fea;
    }
    
    /**
     * @brief Lengthen time to satisfy constraints (same as original ego-planner)
     * 
     * Only modifies middle knots to preserve start/end positions
     */
    void lengthenTime(double ratio) {
        int num1 = 5;
        int num2 = static_cast<int>(u_.rows()) - 1 - 5;
        
        if (num2 <= num1) {
            // Fallback for short trajectories: scale all knots
            interval_ *= ratio;
            for (int i = 0; i <= m_; ++i) {
                u_(i) *= ratio;
            }
            return;
        }
        
        double delta_t = (ratio - 1.0) * (u_(num2) - u_(num1));
        double t_inc = delta_t / double(num2 - num1);
        
        for (int i = num1 + 1; i <= num2; ++i) {
            u_(i) += double(i - num1) * t_inc;
        }
        for (int i = num2 + 1; i < static_cast<int>(u_.rows()); ++i) {
            u_(i) += delta_t;
        }
        
        // Update interval based on new knot spacing
        if (m_ > p_) {
            interval_ = (u_(m_ - p_) - u_(p_)) / (m_ - 2 * p_);
        }
    }
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
private:
    Eigen::MatrixXd control_points_;  // 3 x (n+1) control points
    int p_;                            // B-spline degree
    int n_;                            // n+1 control points
    int m_;                            // m+1 knots
    Eigen::VectorXd u_;               // Knot vector
    double interval_;                  // Knot span
    double limit_vel_ = 1.0;
    double limit_acc_ = 1.0;
    double feasibility_tolerance_ = 0.0;
};

/**
 * @brief Generate initial B-spline control points from path
 * @param path Sequence of waypoints (3 x N)
 * @param ts Desired time interval between control points
 * @param ctrl_pts Output control points
 * 
 * For a path with K+2 points and boundary vel/acc, generates K+6 control points
 * Here we use a simplified version: uniformly space control points along path
 */
inline Eigen::MatrixXd pathToControlPoints(
    const std::vector<Eigen::Vector3d>& path,
    int num_ctrl_pts,
    int order = 3
) {
    if (path.size() < 2) {
        return Eigen::MatrixXd();
    }
    
    // Compute cumulative arc length
    std::vector<double> arc_len(path.size());
    arc_len[0] = 0.0;
    for (size_t i = 1; i < path.size(); ++i) {
        arc_len[i] = arc_len[i-1] + (path[i] - path[i-1]).norm();
    }
    double total_len = arc_len.back();
    
    // Sample control points uniformly along arc length
    Eigen::MatrixXd ctrl_pts(3, num_ctrl_pts);
    for (int i = 0; i < num_ctrl_pts; ++i) {
        double target_s = (i * total_len) / (num_ctrl_pts - 1);
        
        // Find segment
        int seg = 0;
        for (size_t j = 1; j < path.size(); ++j) {
            if (arc_len[j] >= target_s) {
                seg = j - 1;
                break;
            }
            seg = j - 1;
        }
        
        // Interpolate
        double seg_len = arc_len[seg + 1] - arc_len[seg];
        double alpha = 0.0;
        if (seg_len > 1e-6) {
            alpha = (target_s - arc_len[seg]) / seg_len;
        }
        ctrl_pts.col(i) = (1.0 - alpha) * path[seg] + alpha * path[seg + 1];
    }
    
    return ctrl_pts;
}

/**
 * @brief Generate straight-line initial control points
 * @param start Start position
 * @param end End position
 * @param num_ctrl_pts Number of control points
 */
inline Eigen::MatrixXd straightLineControlPoints(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& end,
    int num_ctrl_pts
) {
    Eigen::MatrixXd ctrl_pts(3, num_ctrl_pts);
    for (int i = 0; i < num_ctrl_pts; ++i) {
        double alpha = static_cast<double>(i) / (num_ctrl_pts - 1);
        ctrl_pts.col(i) = (1.0 - alpha) * start + alpha * end;
    }
    return ctrl_pts;
}

/**
 * @brief Generate B-spline control points from path points with velocity/acceleration constraints
 * 
 * This is the key function from original ego-planner that ensures smooth trajectory
 * at start and end points by satisfying velocity and acceleration constraints.
 * 
 * The function solves a least-squares problem:
 *   min ||A * ctrl_pts - b||^2
 * where A encodes the B-spline basis functions and derivative constraints.
 * 
 * @param ts Time interval between control points
 * @param point_set K path points that the B-spline should approximately pass through
 * @param start_end_derivatives [start_vel, end_vel, start_acc, end_acc] (4 vectors)
 * @param ctrl_pts Output: (K+2) control points
 */
inline void parameterizeToBspline(
    double ts,
    const std::vector<Eigen::Vector3d>& point_set,
    const std::vector<Eigen::Vector3d>& start_end_derivatives,
    Eigen::MatrixXd& ctrl_pts
) {
    if (ts <= 0) {
        std::cerr << "[B-spline]: time step error." << std::endl;
        return;
    }
    
    int K = static_cast<int>(point_set.size());
    if (K <= 3) {
        std::cerr << "[B-spline]: point set have only " << K << " points." << std::endl;
        return;
    }
    
    if (start_end_derivatives.size() != 4) {
        std::cerr << "[B-spline]: derivatives error (need 4, got " 
                  << start_end_derivatives.size() << ")." << std::endl;
        return;
    }
    
    // B-spline basis coefficients for cubic B-spline:
    // Position: (1/6) * [1, 4, 1]
    // Velocity: (1/2/ts) * [-1, 0, 1]
    // Acceleration: (1/ts^2) * [1, -2, 1]
    Eigen::Vector3d prow(3), vrow(3), arow(3);
    prow << 1.0, 4.0, 1.0;
    vrow << -1.0, 0.0, 1.0;
    arow << 1.0, -2.0, 1.0;
    
    // Build matrix A (K+4 rows, K+2 columns)
    // First K rows: position constraints from point_set
    // Next 2 rows: velocity constraints (start and end)
    // Last 2 rows: acceleration constraints (start and end)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K + 4, K + 2);
    
    // Position constraints: B(t_i) ≈ point_set[i]
    for (int i = 0; i < K; ++i) {
        A.block(i, i, 1, 3) = (1.0 / 6.0) * prow.transpose();
    }
    
    // Velocity constraints at start and end
    A.block(K, 0, 1, 3) = (1.0 / (2.0 * ts)) * vrow.transpose();
    A.block(K + 1, K - 1, 1, 3) = (1.0 / (2.0 * ts)) * vrow.transpose();
    
    // Acceleration constraints at start and end
    A.block(K + 2, 0, 1, 3) = (1.0 / (ts * ts)) * arow.transpose();
    A.block(K + 3, K - 1, 1, 3) = (1.0 / (ts * ts)) * arow.transpose();
    
    // Build right-hand side b
    Eigen::VectorXd bx(K + 4), by(K + 4), bz(K + 4);
    for (int i = 0; i < K; ++i) {
        bx(i) = point_set[i](0);
        by(i) = point_set[i](1);
        bz(i) = point_set[i](2);
    }
    
    // Derivative constraints: [start_vel, end_vel, start_acc, end_acc]
    bx(K) = start_end_derivatives[0](0);     // start_vel.x
    by(K) = start_end_derivatives[0](1);
    bz(K) = start_end_derivatives[0](2);
    
    bx(K + 1) = start_end_derivatives[1](0); // end_vel.x
    by(K + 1) = start_end_derivatives[1](1);
    bz(K + 1) = start_end_derivatives[1](2);
    
    bx(K + 2) = start_end_derivatives[2](0); // start_acc.x
    by(K + 2) = start_end_derivatives[2](1);
    bz(K + 2) = start_end_derivatives[2](2);
    
    bx(K + 3) = start_end_derivatives[3](0); // end_acc.x
    by(K + 3) = start_end_derivatives[3](1);
    bz(K + 3) = start_end_derivatives[3](2);
    
    // Solve least squares: A * ctrl_pts = b
    Eigen::VectorXd px = A.colPivHouseholderQr().solve(bx);
    Eigen::VectorXd py = A.colPivHouseholderQr().solve(by);
    Eigen::VectorXd pz = A.colPivHouseholderQr().solve(bz);
    
    // Output control points (3 x (K+2))
    ctrl_pts.resize(3, K + 2);
    ctrl_pts.row(0) = px.transpose();
    ctrl_pts.row(1) = py.transpose();
    ctrl_pts.row(2) = pz.transpose();
}

/**
 * @brief Convenience wrapper for parameterizeToBspline with separate start/end derivatives
 */
inline Eigen::MatrixXd parameterizeToBsplineSimple(
    double ts,
    const std::vector<Eigen::Vector3d>& point_set,
    const Eigen::Vector3d& start_vel,
    const Eigen::Vector3d& end_vel,
    const Eigen::Vector3d& start_acc = Eigen::Vector3d::Zero(),
    const Eigen::Vector3d& end_acc = Eigen::Vector3d::Zero()
) {
    std::vector<Eigen::Vector3d> derivatives = {start_vel, end_vel, start_acc, end_acc};
    Eigen::MatrixXd ctrl_pts;
    parameterizeToBspline(ts, point_set, derivatives, ctrl_pts);
    return ctrl_pts;
}

} // namespace ego_planner

#endif // UNIFORM_BSPLINE_HPP



