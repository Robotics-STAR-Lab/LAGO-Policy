/**
 * @file polynomial_traj.hpp
 * @brief Polynomial trajectory generation for ego-planner
 * 
 * This file implements polynomial trajectory generation, essential for
 * smooth replanning with velocity and acceleration continuity.
 * 
 * Ported from ego-planner's polynomial_traj.cpp
 */

#ifndef POLYNOMIAL_TRAJ_HPP
#define POLYNOMIAL_TRAJ_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ego_planner {

/**
 * @brief Polynomial trajectory class
 * 
 * Represents a trajectory as a sequence of polynomial segments.
 * Each segment is a 5th-order polynomial that can satisfy boundary conditions
 * (position, velocity, acceleration) at start and end points.
 */
class PolynomialTraj {
public:
    struct Segment {
        std::vector<double> cx, cy, cz;  // Polynomial coefficients [c5, c4, c3, c2, c1, c0]
        double duration;
        
        Segment() : duration(0) {
            cx.resize(6, 0);
            cy.resize(6, 0);
            cz.resize(6, 0);
        }
    };

private:
    std::vector<Segment> segments_;
    std::vector<double> times_;  // Cumulative times
    double total_time_;

public:
    PolynomialTraj() : total_time_(0) {}
    
    /**
     * @brief Add a polynomial segment
     * @param cx, cy, cz Polynomial coefficients [c5, c4, c3, c2, c1, c0]
     *        p(t) = c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0
     * @param duration Duration of this segment
     */
    void addSegment(const std::vector<double>& cx, 
                    const std::vector<double>& cy,
                    const std::vector<double>& cz,
                    double duration) {
        Segment seg;
        seg.cx = cx;
        seg.cy = cy;
        seg.cz = cz;
        seg.duration = duration;
        segments_.push_back(seg);
        
        times_.push_back(total_time_);
        total_time_ += duration;
    }
    
    /**
     * @brief Evaluate position at time t
     */
    Eigen::Vector3d evaluate(double t) const {
        if (segments_.empty()) return Eigen::Vector3d::Zero();
        
        // Clamp time
        t = std::max(0.0, std::min(t, total_time_));
        
        // Find segment
        int seg_idx = 0;
        double local_t = t;
        for (size_t i = 0; i < segments_.size(); ++i) {
            if (t <= times_[i] + segments_[i].duration || i == segments_.size() - 1) {
                seg_idx = i;
                local_t = t - times_[i];
                break;
            }
        }
        
        const Segment& seg = segments_[seg_idx];
        local_t = std::max(0.0, std::min(local_t, seg.duration));
        
        // Evaluate polynomial: p(t) = c5*t^5 + c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0
        Eigen::Vector3d pos;
        pos(0) = evaluatePoly(seg.cx, local_t);
        pos(1) = evaluatePoly(seg.cy, local_t);
        pos(2) = evaluatePoly(seg.cz, local_t);
        
        return pos;
    }
    
    /**
     * @brief Evaluate velocity at time t
     */
    Eigen::Vector3d evaluateVel(double t) const {
        if (segments_.empty()) return Eigen::Vector3d::Zero();
        
        t = std::max(0.0, std::min(t, total_time_));
        
        int seg_idx = 0;
        double local_t = t;
        for (size_t i = 0; i < segments_.size(); ++i) {
            if (t <= times_[i] + segments_[i].duration || i == segments_.size() - 1) {
                seg_idx = i;
                local_t = t - times_[i];
                break;
            }
        }
        
        const Segment& seg = segments_[seg_idx];
        local_t = std::max(0.0, std::min(local_t, seg.duration));
        
        // Velocity: derivative of position
        // v(t) = 5*c5*t^4 + 4*c4*t^3 + 3*c3*t^2 + 2*c2*t + c1
        Eigen::Vector3d vel;
        vel(0) = evaluatePolyDeriv(seg.cx, local_t);
        vel(1) = evaluatePolyDeriv(seg.cy, local_t);
        vel(2) = evaluatePolyDeriv(seg.cz, local_t);
        
        return vel;
    }
    
    /**
     * @brief Evaluate acceleration at time t
     */
    Eigen::Vector3d evaluateAcc(double t) const {
        if (segments_.empty()) return Eigen::Vector3d::Zero();
        
        t = std::max(0.0, std::min(t, total_time_));
        
        int seg_idx = 0;
        double local_t = t;
        for (size_t i = 0; i < segments_.size(); ++i) {
            if (t <= times_[i] + segments_[i].duration || i == segments_.size() - 1) {
                seg_idx = i;
                local_t = t - times_[i];
                break;
            }
        }
        
        const Segment& seg = segments_[seg_idx];
        local_t = std::max(0.0, std::min(local_t, seg.duration));
        
        // Acceleration: second derivative
        // a(t) = 20*c5*t^3 + 12*c4*t^2 + 6*c3*t + 2*c2
        Eigen::Vector3d acc;
        acc(0) = evaluatePolyDeriv2(seg.cx, local_t);
        acc(1) = evaluatePolyDeriv2(seg.cy, local_t);
        acc(2) = evaluatePolyDeriv2(seg.cz, local_t);
        
        return acc;
    }
    
    double getTotalTime() const { return total_time_; }
    
    /**
     * @brief Generate a single-segment polynomial trajectory satisfying boundary conditions
     * 
     * This is the key function from ego-planner for smooth replanning.
     * It generates a 5th-order polynomial that exactly satisfies:
     * - Start position, velocity, acceleration
     * - End position, velocity, acceleration
     * 
     * @param start_pt Start position
     * @param start_vel Start velocity
     * @param start_acc Start acceleration
     * @param end_pt End position
     * @param end_vel End velocity
     * @param end_acc End acceleration
     * @param duration Duration of the trajectory
     * @return PolynomialTraj with one segment
     */
    static PolynomialTraj one_segment_traj_gen(
        const Eigen::Vector3d& start_pt, const Eigen::Vector3d& start_vel, const Eigen::Vector3d& start_acc,
        const Eigen::Vector3d& end_pt, const Eigen::Vector3d& end_vel, const Eigen::Vector3d& end_acc,
        double duration
    ) {
        // Build constraint matrix C
        // For a 5th order polynomial p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        // 
        // Constraints at t=0:
        //   p(0) = c0 = start_pt
        //   p'(0) = c1 = start_vel
        //   p''(0) = 2*c2 = start_acc
        //
        // Constraints at t=T (duration):
        //   p(T) = c0 + c1*T + c2*T^2 + c3*T^3 + c4*T^4 + c5*T^5 = end_pt
        //   p'(T) = c1 + 2*c2*T + 3*c3*T^2 + 4*c4*T^3 + 5*c5*T^4 = end_vel
        //   p''(T) = 2*c2 + 6*c3*T + 12*c4*T^2 + 20*c5*T^3 = end_acc
        
        double t = duration;
        
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(6, 6);
        
        // Constraint at t=0: p(0) = c0
        C(0, 5) = 1;
        
        // Constraint at t=0: p'(0) = c1
        C(1, 4) = 1;
        
        // Constraint at t=0: p''(0) = 2*c2
        C(2, 3) = 2;
        
        // Constraint at t=T: p(T)
        Eigen::VectorXd Crow(6);
        Crow << std::pow(t, 5), std::pow(t, 4), std::pow(t, 3), std::pow(t, 2), t, 1;
        C.row(3) = Crow.transpose();
        
        // Constraint at t=T: p'(T)
        Crow << 5*std::pow(t, 4), 4*std::pow(t, 3), 3*std::pow(t, 2), 2*t, 1, 0;
        C.row(4) = Crow.transpose();
        
        // Constraint at t=T: p''(T)
        Crow << 20*std::pow(t, 3), 12*std::pow(t, 2), 6*t, 2, 0, 0;
        C.row(5) = Crow.transpose();
        
        // Right-hand side
        Eigen::VectorXd Bx(6), By(6), Bz(6);
        Bx << start_pt(0), start_vel(0), start_acc(0), end_pt(0), end_vel(0), end_acc(0);
        By << start_pt(1), start_vel(1), start_acc(1), end_pt(1), end_vel(1), end_acc(1);
        Bz << start_pt(2), start_vel(2), start_acc(2), end_pt(2), end_vel(2), end_acc(2);
        
        // Solve for coefficients
        Eigen::VectorXd Cofx = C.colPivHouseholderQr().solve(Bx);
        Eigen::VectorXd Cofy = C.colPivHouseholderQr().solve(By);
        Eigen::VectorXd Cofz = C.colPivHouseholderQr().solve(Bz);
        
        // Convert to std::vector (coefficients are [c5, c4, c3, c2, c1, c0])
        std::vector<double> cx(6), cy(6), cz(6);
        for (int i = 0; i < 6; ++i) {
            cx[i] = Cofx(i);
            cy[i] = Cofy(i);
            cz[i] = Cofz(i);
        }
        
        PolynomialTraj poly_traj;
        poly_traj.addSegment(cx, cy, cz, duration);
        
        return poly_traj;
    }
    
    /**
     * @brief Sample the trajectory at regular intervals
     * @param dt Time step for sampling
     * @return Vector of sampled positions
     */
    std::vector<Eigen::Vector3d> sampleTrajectory(double dt) const {
        std::vector<Eigen::Vector3d> samples;
        for (double t = 0; t <= total_time_; t += dt) {
            samples.push_back(evaluate(t));
        }
        // Ensure we include the final point
        if (!samples.empty() && (total_time_ - (samples.size() - 1) * dt) > 1e-6) {
            samples.push_back(evaluate(total_time_));
        }
        return samples;
    }

private:
    // Evaluate polynomial: p(t) = c[0]*t^5 + c[1]*t^4 + c[2]*t^3 + c[3]*t^2 + c[4]*t + c[5]
    static double evaluatePoly(const std::vector<double>& c, double t) {
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        double t5 = t4 * t;
        return c[0]*t5 + c[1]*t4 + c[2]*t3 + c[3]*t2 + c[4]*t + c[5];
    }
    
    // First derivative: v(t) = 5*c[0]*t^4 + 4*c[1]*t^3 + 3*c[2]*t^2 + 2*c[3]*t + c[4]
    static double evaluatePolyDeriv(const std::vector<double>& c, double t) {
        double t2 = t * t;
        double t3 = t2 * t;
        double t4 = t3 * t;
        return 5*c[0]*t4 + 4*c[1]*t3 + 3*c[2]*t2 + 2*c[3]*t + c[4];
    }
    
    // Second derivative: a(t) = 20*c[0]*t^3 + 12*c[1]*t^2 + 6*c[2]*t + 2*c[3]
    static double evaluatePolyDeriv2(const std::vector<double>& c, double t) {
        double t2 = t * t;
        double t3 = t2 * t;
        return 20*c[0]*t3 + 12*c[1]*t2 + 6*c[2]*t + 2*c[3];
    }
};

} // namespace ego_planner

#endif // POLYNOMIAL_TRAJ_HPP


