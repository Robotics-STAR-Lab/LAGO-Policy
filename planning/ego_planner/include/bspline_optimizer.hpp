/**
 * @file bspline_optimizer.hpp
 * @brief B-spline trajectory optimizer for robotic arm obstacle avoidance
 * 
 * Standalone implementation based on ego-planner, without ROS dependency.
 * 
 * Two-stage optimization:
 * 1. Rebound optimization: smoothness + collision + feasibility
 * 2. Refine optimization: smoothness + feasibility + fitness (curve fitting)
 * 
 * Key parameters:
 * - Nc (num_ctrl_pts): Number of control points
 * - ts (bspline_interval): Time interval between control points
 */

#ifndef BSPLINE_OPTIMIZER_HPP
#define BSPLINE_OPTIMIZER_HPP

#include <Eigen/Dense>
#include <vector>
#include <chrono>
#include <iomanip>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include "uniform_bspline.hpp"
#include "astar_searcher.hpp"
#include "grid_map.hpp"
#include "lbfgs.hpp"

namespace ego_planner {

// Helper functions
inline Eigen::Vector3d aabbCenter(const AABB& box) {
    return 0.5 * (box.min_corner + box.max_corner);
}

inline Eigen::Vector3d aabbHalfSize(const AABB& box) {
    return 0.5 * (box.max_corner - box.min_corner);
}

inline double signedDistanceToAABB(const Eigen::Vector3d& point, const AABB& box) {
    Eigen::Vector3d center = aabbCenter(box);
    Eigen::Vector3d half = aabbHalfSize(box);
    Eigen::Vector3d d = (point - center).cwiseAbs() - half;
    double outside_dist = d.cwiseMax(0.0).norm();
    double inside_dist = std::min(d.maxCoeff(), 0.0);
    return outside_dist + inside_dist;
}

/**
 * @brief Control points with collision direction info
 */
struct ControlPoints {
    double clearance = 0.05;
    int size = 0;
    Eigen::MatrixXd points;  // 3 x N
    std::vector<std::vector<Eigen::Vector3d>> base_point;  // Collision surface points
    std::vector<std::vector<Eigen::Vector3d>> direction;   // Push directions (normalized)
    std::vector<bool> flag_temp;
    
    void resize(int n) {
        size = n;
        points.resize(3, n);
        base_point.resize(n);
        direction.resize(n);
        flag_temp.resize(n);
        clearDirections();
    }
    
    void clearDirections() {
        for (int i = 0; i < size; ++i) {
            base_point[i].clear();
            direction[i].clear();
            flag_temp[i] = false;
        }
    }
};

/**
 * @brief B-spline trajectory optimizer (ego-planner style)
 */
class BsplineOptimizer {
public:
    BsplineOptimizer() {
        order_ = 3;
        lambda_smooth_ = 1.0;
        lambda_collision_ = 10.0;
        lambda_feasibility_ = 0.1;
        lambda_fitness_ = 1.0;
        dist0_ = 0.05;
        max_vel_ = 0.5;
        max_acc_ = 2.0;
        astar_ = std::make_unique<AStarSearcher>();
    }
    
    ~BsplineOptimizer() = default;
    
    // ==================== Parameter Settings ====================
    
    void setWeights(double smooth, double collision, double feasibility, double fitness = 1.0) {
        lambda_smooth_ = smooth;
        lambda_collision_ = collision;
        lambda_feasibility_ = feasibility;
        lambda_fitness_ = fitness;
    }
    
    void setPhysicalLimits(double max_vel, double max_acc, double safe_dist) {
        max_vel_ = max_vel;
        max_acc_ = max_acc;
        dist0_ = safe_dist;
    }
    
    /**
     * @brief Set ground/table level constraint for A* search
     * A* will not search below this Z level, forcing upward avoidance for robotic arm
     */
    void setGroundLevel(double ground_z) {
        ground_z_limit_ = ground_z;
    }
    double getGroundLevel() const { return ground_z_limit_; }
    
    /**
     * @brief Set fixed workspace bounds (avoids repeated GridMap initialization)
     * 
     * When set, the optimizer will use this fixed workspace for all optimizations,
     * which significantly speeds up repeated calls by avoiding GridMap reinitialization.
     * 
     * @param min_pt Minimum corner of workspace (x, y, z)
     * @param max_pt Maximum corner of workspace (x, y, z)
     */
    void setWorkspace(const Eigen::Vector3d& min_pt, const Eigen::Vector3d& max_pt) {
        workspace_min_ = min_pt;
        workspace_max_ = max_pt;
        use_fixed_workspace_ = true;
        workspace_initialized_ = false;  // Force reinitialization with new bounds
    }
    
    /**
     * @brief Clear fixed workspace (use auto-computed bounds)
     */
    void clearWorkspace() {
        use_fixed_workspace_ = false;
        workspace_initialized_ = false;
    }
    
    bool hasFixedWorkspace() const { return use_fixed_workspace_; }
    
    /**
     * @brief Enable/disable preference for vertical (Z-axis) avoidance paths
     * When enabled, A* will strongly prefer paths that go UP rather than sideways.
     * This is the recommended setting for robotic arm obstacle avoidance.
     * 
     * @param prefer Enable vertical preference (default: true)
     * @param discount Cost discount for Z movement (0.3 = 70% discount, lower = stronger preference)
     */
    void setPreferVertical(bool prefer, double discount = 0.3) {
        prefer_vertical_ = prefer;
        vertical_discount_ = discount;
        if (astar_) {
            astar_->setPreferVertical(prefer, discount);
        }
    }
    bool getPreferVertical() const { return prefer_vertical_; }
    
    /**
     * @brief Enable/disable verbose debug output (default: off for performance)
     */
    void setVerbose(bool verbose) { verbose_ = verbose; }
    bool getVerbose() const { return verbose_; }
    
    void setOrder(int order) { order_ = order; }
    int getOrder() const { return order_; }
    
    double getMaxVel() const { return max_vel_; }
    double getMaxAcc() const { return max_acc_; }
    double getSafeDist() const { return dist0_; }
    
    // ==================== Obstacle Management ====================
    
    void clearObstacles() { 
        obstacles_.clear(); 
        grid_map_initialized_ = false;
    }
    
    void addObstacle(const AABB& obs) { 
        obstacles_.push_back(obs); 
        grid_map_initialized_ = false;
    }
    
    void addObstacle(const Eigen::Vector3d& min_corner, const Eigen::Vector3d& max_corner) {
        obstacles_.emplace_back(min_corner, max_corner);
        grid_map_initialized_ = false;
    }
    
    void setObstacles(const std::vector<AABB>& obstacles) { 
        obstacles_ = obstacles; 
        grid_map_initialized_ = false;
    }
    
    const std::vector<AABB>& getObstacles() const { return obstacles_; }
    
    /**
     * @brief Initialize GridMap for fast collision detection
     * Call this after adding all obstacles if you want to use isCollision()
     * outside of optimization.
     */
    void initGridMap() {
        if (obstacles_.empty()) return;
        
        // Compute bounds from obstacles
        Eigen::Vector3d min_pt = obstacles_[0].min_corner;
        Eigen::Vector3d max_pt = obstacles_[0].max_corner;
        for (const auto& obs : obstacles_) {
            min_pt = min_pt.cwiseMin(obs.min_corner);
            max_pt = max_pt.cwiseMax(obs.max_corner);
        }
        
        // Also include control points if available
        if (cps_.size > 0) {
            min_pt = min_pt.cwiseMin(cps_.points.rowwise().minCoeff());
            max_pt = max_pt.cwiseMax(cps_.points.rowwise().maxCoeff());
        }
        
        // Expand significantly (same as setupAStarSearcher)
        double expand = 0.5;
        min_pt -= Eigen::Vector3d::Constant(expand);
        max_pt += Eigen::Vector3d::Constant(expand);
        
        double resolution = 0.01;
        Eigen::Vector3d size_d = (max_pt - min_pt) / resolution;
        Eigen::Vector3i size(
            std::max(30, std::min(300, (int)std::ceil(size_d(0)))),
            std::max(30, std::min(300, (int)std::ceil(size_d(1)))),
            std::max(30, std::min(300, (int)std::ceil(size_d(2))))
        );
        
        if (!grid_map_) {
            grid_map_ = std::make_unique<GridMap>();
        }
        grid_map_->init(resolution, min_pt, size);
        grid_map_->setObstacles(obstacles_, dist0_);
        grid_map_initialized_ = true;
    }
    
    // ==================== Main Optimization API ====================
    
    /**
     * @brief Optimize trajectory from start to goal
     * @param start Start position
     * @param goal Goal position
     * @param num_ctrl_pts Number of control points (Nc)
     * @param ts Time interval between control points
     * @param max_iterations Maximum iterations per optimization
     * @return Optimized B-spline trajectory
     */
    UniformBspline optimizeTrajectory(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& goal,
        int num_ctrl_pts = 15,
        double ts = 0.1,
        int max_iterations = 100
    ) {
        // Generate path points from start to goal (same as original ego-planner approach)
        // We need num_ctrl_pts - 2 interior points for parameterizeToBspline
        // because it produces K + 2 control points from K path points
        int K = num_ctrl_pts - 2;
        if (K < 4) K = 4;  // Minimum points needed
        
        std::vector<Eigen::Vector3d> path_points(K);
        for (int i = 0; i < K; ++i) {
            double alpha = static_cast<double>(i) / (K - 1);
            path_points[i] = (1.0 - alpha) * start + alpha * goal;
        }
        
        // Use parameterizeToBspline to generate control points
        // Start and end with zero velocity and acceleration (static endpoints)
        std::vector<Eigen::Vector3d> start_end_derivatives = {
            Eigen::Vector3d::Zero(),  // start velocity
            Eigen::Vector3d::Zero(),  // end velocity
            Eigen::Vector3d::Zero(),  // start acceleration
            Eigen::Vector3d::Zero()   // end acceleration
        };
        
        Eigen::MatrixXd init_pts;
        parameterizeToBspline(ts, path_points, start_end_derivatives, init_pts);
        
        return optimizeTrajectory(init_pts, ts, max_iterations);
    }
    
    /**
     * @brief Optimize trajectory from initial control points
     */
    UniformBspline optimizeTrajectory(
        const Eigen::MatrixXd& init_ctrl_pts,
        double ts,
        int max_iterations = 100
    ) {
        bspline_interval_ = ts;
        cps_.resize(init_ctrl_pts.cols());
        cps_.points = init_ctrl_pts;
        cps_.clearance = dist0_;
        
        auto t0 = std::chrono::high_resolution_clock::now();
        setupAStarSearcher();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // Stage 1: Rebound optimization (collision-free)
        initControlPointsWithAStar();
        auto t2 = std::chrono::high_resolution_clock::now();
        bool rebound_success = reboundOptimize(max_iterations);
        auto t3 = std::chrono::high_resolution_clock::now();
        
        double setup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double astar_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double lbfgs_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        if (verbose_) {
            std::cout << "[Timing] setup=" << std::fixed << std::setprecision(2) << setup_ms 
                      << "ms, A*=" << astar_ms << "ms, L-BFGS=" << lbfgs_ms << "ms" << std::endl;
        }
        
        // Stage 1 may not fully converge but still produce collision-free result
        // through A* path initialization
        
        // Stage 2: Time reallocation + Refine optimization
        UniformBspline traj(cps_.points, order_, bspline_interval_);
        traj.setPhysicalLimits(max_vel_, max_acc_);
        
        double ratio;
        if (!traj.checkFeasibility(ratio)) {
            // Time reallocation
            double new_ts = bspline_interval_ * ratio * 1.1;  // 10% margin
            ref_pts_.clear();
            for (int i = 0; i < cps_.size; ++i) {
                ref_pts_.push_back(cps_.points.col(i));
            }
            
            bspline_interval_ = new_ts;
            refineOptimize(max_iterations);
        }
        
        return UniformBspline(cps_.points, order_, bspline_interval_);
    }
    
    /**
     * @brief Optimize given B-spline with rebound only (Stage 1)
     */
    bool BsplineOptimizeTrajRebound(Eigen::MatrixXd& optimal_points, double ts) {
        bspline_interval_ = ts;
        cps_.resize(optimal_points.cols());
        cps_.points = optimal_points;
        cps_.clearance = dist0_;
        
        setupAStarSearcher();
        initControlPointsWithAStar();
        bool success = reboundOptimize(200);
        
        optimal_points = cps_.points;
        return success;
    }
    
    /**
     * @brief Optimize with refine (Stage 2) after time reallocation
     */
    bool BsplineOptimizeTrajRefine(const Eigen::MatrixXd& init_points, double ts,
                                    Eigen::MatrixXd& optimal_points) {
        cps_.resize(init_points.cols());
        cps_.points = init_points;
        bspline_interval_ = ts;
        
        ref_pts_.clear();
        for (int i = 0; i < cps_.size; ++i) {
            ref_pts_.push_back(init_points.col(i));
        }
        
        bool success = refineOptimize(200);
        optimal_points = cps_.points;
        return success;
    }
    
    /**
     * @brief Check collision using GridMap (O(1)) or AABB list (O(n))
     */
    bool isCollision(const Eigen::Vector3d& point) const {
        // Use GridMap for O(1) collision detection if available and initialized
        if (use_grid_map_ && grid_map_ && grid_map_initialized_) {
            return grid_map_->getInflateOccupancy(point);
        }
        
        // Fallback to AABB list traversal (O(n))
        for (const auto& obs : obstacles_) {
            if (signedDistanceToAABB(point, obs) < dist0_) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Get minimum distance to obstacles
     */
    double getMinDistance(const Eigen::Vector3d& point) const {
        // GridMap can provide distance, but for accuracy use AABB directly
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& obs : obstacles_) {
            min_dist = std::min(min_dist, signedDistanceToAABB(point, obs));
        }
        return min_dist;
    }
    
    /**
     * @brief Enable/disable GridMap usage
     */
    void setUseGridMap(bool use) { use_grid_map_ = use; }
    bool getUseGridMap() const { return use_grid_map_; }
    
    Eigen::MatrixXd getControlPoints() const { return cps_.points; }
    
private:
    // Parameters
    int order_;
    double lambda_smooth_;
    double lambda_collision_;
    double lambda_feasibility_;
    double lambda_fitness_;
    double dist0_;
    double max_vel_;
    double max_acc_;
    double bspline_interval_;
    double ground_z_limit_ = 0.05;  // Ground/table surface level (default 5cm above origin)
    bool prefer_vertical_ = true;   // Prefer vertical (Z) avoidance paths (default: enabled)
    double vertical_discount_ = 0.3;  // Cost discount for Z movement (0.3 = 70% cheaper)
    
    // Fixed workspace for avoiding repeated GridMap initialization
    bool use_fixed_workspace_ = false;
    Eigen::Vector3d workspace_min_ = Eigen::Vector3d(0, -0.5, 0);
    Eigen::Vector3d workspace_max_ = Eigen::Vector3d(1, 0.5, 0.5);
    bool workspace_initialized_ = false;
    
    // State
    ControlPoints cps_;
    std::vector<AABB> obstacles_;
    std::unique_ptr<AStarSearcher> astar_;
    std::unique_ptr<GridMap> grid_map_;  // O(1) collision detection
    std::vector<Eigen::Vector3d> ref_pts_;  // Reference points for refine stage
    int iter_num_;
    double new_lambda2_;
    bool use_grid_map_ = true;  // Enabled by default - matches original ego-planner
    bool grid_map_initialized_ = false;  // Track if GridMap has been initialized
    bool verbose_ = false;  // Debug output (default: off for performance)
    
    enum ForceStopType { DONT_STOP, STOP_FOR_REBOUND, STOP_FOR_ERROR } force_stop_type_;
    
    void setupAStarSearcher() {
        Eigen::Vector3d min_pt, max_pt;
        
        // Use fixed workspace if set (avoids repeated GridMap initialization)
        if (use_fixed_workspace_) {
            min_pt = workspace_min_;
            max_pt = workspace_max_;
            
            // If already initialized with fixed workspace, skip reinitialization
            if (workspace_initialized_ && grid_map_initialized_) {
                // Just update A* settings (cheap operation)
                astar_->setPreferVertical(prefer_vertical_, vertical_discount_);
                return;
            }
        } else {
            // Auto-compute workspace from trajectory and obstacles
            min_pt = cps_.points.rowwise().minCoeff();
            max_pt = cps_.points.rowwise().maxCoeff();
            
            for (const auto& obs : obstacles_) {
                min_pt = min_pt.cwiseMin(obs.min_corner);
                max_pt = max_pt.cwiseMax(obs.max_corner);
            }
            
            // CRITICAL: For robotic arm, we need to:
            // 1. Expand MORE in Z+ direction for upward avoidance
            // 2. NOT expand downward much - the table is there!
            double expand_xy = 0.3;  // Horizontal expansion (increased)
            double expand_z_up = 0.4;  // Upward expansion (more for avoidance)
            
            min_pt(0) -= expand_xy;
            min_pt(1) -= expand_xy;
            max_pt(0) += expand_xy;
            max_pt(1) += expand_xy;
            max_pt(2) += expand_z_up;  // More space upward!
            
            // Ensure max Z is well above any obstacle top surface
            for (const auto& obs : obstacles_) {
                if (max_pt(2) < obs.max_corner(2) + 0.2) {
                    max_pt(2) = obs.max_corner(2) + 0.2;
                }
            }
        }
        
        // CRITICAL: Set ground plane constraint for robotic arm
        double ground_z = ground_z_limit_;
        if (min_pt(2) < ground_z) {
            min_pt(2) = ground_z;
        }
        
        // Ensure reasonable Z range for upward avoidance
        double min_z_range = 0.2;  // At least 20cm above ground
        if (max_pt(2) - min_pt(2) < min_z_range) {
            max_pt(2) = min_pt(2) + min_z_range;
        }
        
        // Use fine resolution for A* (2cm) - good balance for robotic arm
        double astar_resolution = 0.02;  // 2cm resolution
        Eigen::Vector3d size_d = (max_pt - min_pt) / astar_resolution;
        Eigen::Vector3i astar_size(
            std::max(20, std::min(150, (int)std::ceil(size_d(0)))),
            std::max(20, std::min(150, (int)std::ceil(size_d(1)))),
            std::max(20, std::min(150, (int)std::ceil(size_d(2))))
        );
        
        if (verbose_) {
            std::cout << "[A*] Workspace: [" << min_pt.transpose() << "] to [" 
                      << max_pt.transpose() << "], ground_z=" << ground_z 
                      << ", resolution=" << astar_resolution 
                      << ", size=" << astar_size.transpose() << std::endl;
        }
        
        astar_->init(astar_resolution, min_pt, astar_size);
        astar_->setObstacles(obstacles_, dist0_ * 1.2);  // Slightly inflated for A*
        astar_->setPreferVertical(prefer_vertical_, vertical_discount_);
        
        // Use same resolution for GridMap collision detection
        double gridmap_resolution = astar_resolution;
        Eigen::Vector3i gridmap_size = astar_size;
        
        // Initialize GridMap for O(1) collision detection
        if (use_grid_map_) {
            if (!grid_map_) {
                grid_map_ = std::make_unique<GridMap>();
            }
            grid_map_->init(gridmap_resolution, min_pt, gridmap_size);
            grid_map_->setObstacles(obstacles_, dist0_);
            grid_map_initialized_ = true;
        }
        
        // Mark workspace as initialized for fixed workspace mode
        if (use_fixed_workspace_) {
            workspace_initialized_ = true;
        }
    }
    
    // ==================== Rebound Optimization (Stage 1) ====================
    
    /**
     * @brief Initialize control points with A* path guidance (similar to original ego-planner)
     * 
     * This function:
     * 1. Finds collision segments in the trajectory
     * 2. Runs A* search to find collision-free paths
     * 3. Computes base_point and direction for each control point based on A* path
     */
    void initControlPointsWithAStar() {
        cps_.clearDirections();
        
        std::vector<std::pair<int, int>> segment_ids;
        findCollisionSegmentsOriginal(segment_ids);
        
        if (segment_ids.empty()) {
            // Double check: use simple collision detection as fallback
            findCollisionSegmentsSimple(segment_ids);
            if (segment_ids.empty()) {
                if (verbose_) std::cout << "[A*] No collision segments found" << std::endl;
                return;
            }
        }
        
        if (verbose_) std::cout << "[A*] Found " << segment_ids.size() << " collision segment(s)" << std::endl;
        
        // Run A* search for each collision segment
        std::vector<std::vector<Eigen::Vector3d>> a_star_paths;
        for (const auto& seg : segment_ids) {
            Eigen::Vector3d in_pt = cps_.points.col(seg.first);
            Eigen::Vector3d out_pt = cps_.points.col(seg.second);
            
            if (verbose_) {
                std::cout << "[A*] Segment [" << seg.first << "," << seg.second 
                          << "]: from " << in_pt.transpose() << " to " << out_pt.transpose() << std::endl;
            }
            
            if (astar_->search(in_pt, out_pt)) {
                const auto& path = astar_->getPath();
                a_star_paths.push_back(path);
                
                if (verbose_) {
                    // Print A* path Z range
                    double min_z = 1e9, max_z = -1e9;
                    for (const auto& pt : path) {
                        min_z = std::min(min_z, pt(2));
                        max_z = std::max(max_z, pt(2));
                    }
                    std::cout << "[A*] Path found: " << path.size() << " points, Z range: [" 
                              << min_z << ", " << max_z << "]" << std::endl;
                }
            } else {
                if (verbose_) std::cout << "[A*] Path search FAILED!" << std::endl;
                // A* failed, use empty path (will trigger fallback)
                a_star_paths.push_back(std::vector<Eigen::Vector3d>());
            }
        }
        
        // Calculate bounds for each segment (same as original)
        std::vector<std::pair<int, int>> bounds(segment_ids.size());
        for (size_t i = 0; i < segment_ids.size(); ++i) {
            int id_low_bound, id_up_bound;
            if (i == 0) {
                id_low_bound = order_;
                if (segment_ids.size() > 1) {
                    id_up_bound = (segment_ids[0].second + segment_ids[1].first - 1) / 2;
                } else {
                    id_up_bound = cps_.size - order_ - 1;
                }
            } else if (i == segment_ids.size() - 1) {
                id_low_bound = (segment_ids[i].first + segment_ids[i-1].second + 1) / 2;
                id_up_bound = cps_.size - order_ - 1;
            } else {
                id_low_bound = (segment_ids[i].first + segment_ids[i-1].second + 1) / 2;
                id_up_bound = (segment_ids[i].second + segment_ids[i+1].first - 1) / 2;
            }
            bounds[i] = std::make_pair(id_low_bound, id_up_bound);
        }
        
        // Assign directions to each segment using A* path (same as original ego-planner)
        for (size_t i = 0; i < segment_ids.size(); ++i) {
            // If A* path is empty or too short, use fallback direction
            if (a_star_paths[i].size() < 2) {
                if (verbose_) std::cout << "[A*] Using fallback direction for segment " << i << std::endl;
                fallbackDirection(segment_ids[i]);
            } else {
                assignDirectionsFromAStarOriginal(segment_ids[i], bounds[i], a_star_paths[i]);
                
                // Verify directions were assigned, use fallback if not
                bool has_direction = false;
                for (int j = segment_ids[i].first; j <= segment_ids[i].second; ++j) {
                    if (!cps_.direction[j].empty()) {
                        has_direction = true;
                        break;
                    }
                }
                if (!has_direction) {
                    if (verbose_) std::cout << "[A*] Direction assignment failed, using fallback for segment " << i << std::endl;
                    fallbackDirection(segment_ids[i]);
                } else if (verbose_) {
                    // Print assigned direction for first collision point
                    for (int j = segment_ids[i].first; j <= segment_ids[i].second; ++j) {
                        if (!cps_.direction[j].empty()) {
                            std::cout << "[A*] Direction at pt " << j << ": " 
                                      << cps_.direction[j].back().transpose() << std::endl;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    /**
     * @brief Simple collision segment detection (fallback method)
     */
    void findCollisionSegmentsSimple(std::vector<std::pair<int, int>>& segments) {
        segments.clear();
        int end_idx = cps_.size - order_;
        int in_id = -1;
        bool in_collision = false;
        
        for (int i = order_; i < end_idx; ++i) {
            bool occ = isCollision(cps_.points.col(i));
            
            // Also check midpoint
            if (i > order_) {
                Eigen::Vector3d mid = 0.5 * (cps_.points.col(i) + cps_.points.col(i-1));
                occ = occ || isCollision(mid);
            }
            
            if (occ && !in_collision) {
                in_id = std::max(order_, i - 1);
                in_collision = true;
            } else if (!occ && in_collision) {
                int out_id = std::min(i, cps_.size - order_ - 1);
                segments.emplace_back(in_id, out_id);
                in_collision = false;
            }
        }
        
        if (in_collision) {
            int out_id = cps_.size - order_ - 1;
            segments.emplace_back(in_id, out_id);
        }
    }
    
    /**
     * @brief Find collision segments (original ego-planner style)
     * 
     * This closely follows the original initControlPoints() logic from ego-planner:
     * - Uses step_size based on grid resolution
     * - Samples points along segment between control points
     * - Uses ENOUGH_INTERVAL to debounce collision state changes
     */
    void findCollisionSegmentsOriginal(std::vector<std::pair<int, int>>& segments) {
        segments.clear();
        
        constexpr int ENOUGH_INTERVAL = 2;
        double resolution = grid_map_ && grid_map_initialized_ ? grid_map_->getResolution() : 0.01;
        
        // Calculate step size (same as original)
        double total_len = (cps_.points.col(0) - cps_.points.col(cps_.size - 1)).norm();
        double segment_len = total_len / std::max(1, cps_.size - 1);
        double step_size = resolution / std::max(0.001, segment_len) / 2;
        step_size = std::max(0.05, std::min(0.5, step_size));  // Clamp to reasonable range
        
        int in_id = -1, out_id = -1;
        int same_occ_state_times = ENOUGH_INTERVAL + 1;
        bool occ, last_occ = false;
        bool flag_got_start = false, flag_got_end = false, flag_got_end_maybe = false;
        
        // Only check closed 2/3 points (same as original)
        int i_end = cps_.size - order_ - (cps_.size - 2 * order_) / 3;
        i_end = std::max(order_, i_end);
        
        for (int i = order_; i <= i_end; ++i) {
            // Sample points along segment between control points
            for (double a = 1.0; a >= 0.0; a -= step_size) {
                Eigen::Vector3d sample_pt = a * cps_.points.col(i - 1) + (1 - a) * cps_.points.col(i);
                occ = isCollision(sample_pt);
                
                if (occ && !last_occ) {
                    if (same_occ_state_times > ENOUGH_INTERVAL || i == order_) {
                        in_id = i - 1;
                        flag_got_start = true;
                    }
                    same_occ_state_times = 0;
                    flag_got_end_maybe = false;
                } else if (!occ && last_occ) {
                    out_id = i;
                    flag_got_end_maybe = true;
                    same_occ_state_times = 0;
                } else {
                    ++same_occ_state_times;
                }
                
                if (flag_got_end_maybe && (same_occ_state_times > ENOUGH_INTERVAL || i == i_end)) {
                    flag_got_end_maybe = false;
                    flag_got_end = true;
                }
                
                last_occ = occ;
                
                if (flag_got_start && flag_got_end) {
                    flag_got_start = false;
                    flag_got_end = false;
                    if (in_id >= 0 && out_id > in_id) {
                        segments.push_back(std::make_pair(in_id, out_id));
                    }
                }
            }
        }
        
        // Handle case where trajectory ends in collision
        if (flag_got_start && !flag_got_end && in_id >= 0) {
            out_id = std::min(i_end, cps_.size - order_ - 1);
            if (out_id > in_id) {
                segments.push_back(std::make_pair(in_id, out_id));
            }
        }
    }
    
    void findCollisionSegments(std::vector<std::pair<int, int>>& segments) {
        // Try original-style implementation first
        findCollisionSegmentsOriginal(segments);
        
        // If no segments found, try simple method as fallback
        if (segments.empty()) {
            findCollisionSegmentsSimple(segments);
        }
    }
    
    /**
     * @brief Fallback direction assignment (push UP for robotic arm)
     * 
     * This is used when A* fails or produces invalid results.
     * For robotic arm obstacle avoidance, the default strategy is to push UP.
     * 
     * Key insight from ego-planner distance cost:
     * - dist = (ctrl_pt - base_pt).dot(direction)
     * - dist_err = clearance - dist
     * - When dist_err > 0: cost is positive, gradient pushes ctrl_pt in +direction
     * 
     * So to push UP:
     * - direction = (0,0,1)
     * - base_pt should be at or below ctrl_pt so dist starts small
     */
    void fallbackDirection(const std::pair<int, int>& seg) {
        if (verbose_) {
            std::cout << "[Fallback] Assigning UP direction for segment [" 
                      << seg.first << "," << seg.second << "]" << std::endl;
        }
        
        for (int j = seg.first; j <= seg.second && j < cps_.size; ++j) {
            Eigen::Vector3d ctrl_pt = cps_.points.col(j);
            
            // For robotic arm, always push UP (Z+)
            Eigen::Vector3d direction = Eigen::Vector3d(0, 0, 1);
            
            // Find the closest obstacle to determine base_pt
            double min_signed_dist = std::numeric_limits<double>::max();
            const AABB* closest_obs = nullptr;
            
            for (const auto& obs : obstacles_) {
                double sd = signedDistanceToAABB(ctrl_pt, obs);
                if (sd < min_signed_dist) {
                    min_signed_dist = sd;
                    closest_obs = &obs;
                }
            }
            
            // Set base_pt to make the distance calculation work correctly
            // We want: dist = (ctrl_pt - base_pt).dot(direction)
            // For pushing UP: base_pt should be at obstacle top surface
            Eigen::Vector3d base_pt = ctrl_pt;
            
            if (closest_obs != nullptr) {
                // Place base_pt at obstacle's top surface (slightly below for clearance)
                // This ensures ctrl_pt below clearance will be pushed up
                base_pt = ctrl_pt;
                base_pt(2) = closest_obs->max_corner(2) + cps_.clearance;
                
                // If ctrl_pt is inside obstacle, adjust base_pt
                if (min_signed_dist < 0) {
                    // Inside obstacle: base_pt at ctrl_pt level so dist=0
                    base_pt = ctrl_pt;
                }
                
                if (verbose_) {
                    std::cout << "[Fallback] pt " << j << ": ctrl=" << ctrl_pt.transpose() 
                              << ", base=" << base_pt.transpose() 
                              << ", dir=" << direction.transpose()
                              << ", signed_dist=" << min_signed_dist << std::endl;
                }
            }
            
            cps_.base_point[j].push_back(base_pt);
            cps_.direction[j].push_back(direction);
        }
    }
    
    /**
     * @brief Assign directions from A* path (original ego-planner style)
     * 
     * This closely follows the original initControlPoints() logic:
     * 1. For each control point, find intersection with A* path
     * 2. Trace from control point to intersection to find base_point
     * 3. Direction points from control point towards intersection (safe region)
     */
    void assignDirectionsFromAStarOriginal(const std::pair<int, int>& seg,
                                           const std::pair<int, int>& bound,
                                           const std::vector<Eigen::Vector3d>& astar_path) {
        if (astar_path.size() < 2) {
            if (verbose_) std::cout << "[AssignDir] A* path too short, using fallback" << std::endl;
            fallbackDirection(seg);
            return;
        }
        
        if (verbose_) {
            std::cout << "[AssignDir] Processing segment [" << seg.first << "," << seg.second 
                      << "], bounds [" << bound.first << "," << bound.second << "]" << std::endl;
        }
        
        double resolution = grid_map_ && grid_map_initialized_ ? grid_map_->getResolution() : 0.02;
        
        // Step 1: Clear temp flags for this segment
        for (int j = bound.first; j <= bound.second && j < cps_.size; ++j) {
            cps_.flag_temp[j] = false;
        }
        
        // Step 2: Find intersection point for control points in collision segment
        int got_intersection_id = -1;
        for (int j = seg.first + 1; j < seg.second && j + 1 < cps_.size; ++j) {
            // Control point tangent direction (same as original)
            Eigen::Vector3d ctrl_pts_law = cps_.points.col(j + 1) - cps_.points.col(j - 1);
            Eigen::Vector3d intersection_point;
            
            // Binary search for intersection with A* path
            int Astar_id = astar_path.size() / 2;
            int last_Astar_id;
            double val = (astar_path[Astar_id] - cps_.points.col(j)).dot(ctrl_pts_law);
            double last_val = val;
            
            while (Astar_id >= 0 && Astar_id < (int)astar_path.size()) {
                last_Astar_id = Astar_id;
                
                if (val >= 0)
                    --Astar_id;
                else
                    ++Astar_id;
                
                if (Astar_id < 0 || Astar_id >= (int)astar_path.size()) break;
                
                val = (astar_path[Astar_id] - cps_.points.col(j)).dot(ctrl_pts_law);
                
                // Found intersection (sign change)
                if (val * last_val <= 0 && (std::abs(val) > 0 || std::abs(last_val) > 0)) {
                    // Linear interpolation to find exact intersection
                    double denom = ctrl_pts_law.dot(astar_path[Astar_id] - astar_path[last_Astar_id]);
                    if (std::abs(denom) > 1e-10) {
                        double t = ctrl_pts_law.dot(cps_.points.col(j) - astar_path[Astar_id]) / denom;
                        intersection_point = astar_path[Astar_id] + 
                            t * (astar_path[Astar_id] - astar_path[last_Astar_id]);
                    } else {
                        intersection_point = astar_path[Astar_id];
                    }
                    
                    got_intersection_id = j;
                    break;
                }
                
                last_val = val;
            }
            
            if (got_intersection_id >= 0) {
                cps_.flag_temp[j] = true;
                double length = (intersection_point - cps_.points.col(j)).norm();
                
                if (length > 1e-5) {
                    // Trace from intersection to control point, find first collision point
                    for (double a = length; a >= 0.0; a -= resolution) {
                        Eigen::Vector3d test_pt = (a / length) * intersection_point + 
                                                  (1 - a / length) * cps_.points.col(j);
                        bool occ = isCollision(test_pt);
                        
                        if (occ || a < resolution) {
                            if (occ) a += resolution;
                            Eigen::Vector3d base_pt = (a / length) * intersection_point + 
                                                      (1 - a / length) * cps_.points.col(j);
                            Eigen::Vector3d dir = (intersection_point - cps_.points.col(j)).normalized();
                            
                            cps_.base_point[j].push_back(base_pt);
                            cps_.direction[j].push_back(dir);
                            
                            if (verbose_) {
                                std::cout << "[AssignDir] pt " << j << ": intersection=" 
                                          << intersection_point.transpose() 
                                          << ", dir=" << dir.transpose() << std::endl;
                            }
                            break;
                        }
                    }
                }
            }
        }
        
        // Corner case: segment length is too short (same as original)
        if (seg.second - seg.first == 1) {
            Eigen::Vector3d ctrl_pts_law = cps_.points.col(seg.second) - cps_.points.col(seg.first);
            Eigen::Vector3d middle_point = 0.5 * (cps_.points.col(seg.second) + cps_.points.col(seg.first));
            Eigen::Vector3d intersection_point;
            
            int Astar_id = astar_path.size() / 2;
            int last_Astar_id;
            double val = (astar_path[Astar_id] - middle_point).dot(ctrl_pts_law);
            double last_val = val;
            
            while (Astar_id >= 0 && Astar_id < (int)astar_path.size()) {
                last_Astar_id = Astar_id;
                
                if (val >= 0)
                    --Astar_id;
                else
                    ++Astar_id;
                
                if (Astar_id < 0 || Astar_id >= (int)astar_path.size()) break;
                
                val = (astar_path[Astar_id] - middle_point).dot(ctrl_pts_law);
                
                if (val * last_val <= 0 && (std::abs(val) > 0 || std::abs(last_val) > 0)) {
                    double denom = ctrl_pts_law.dot(astar_path[Astar_id] - astar_path[last_Astar_id]);
                    if (std::abs(denom) > 1e-10) {
                        double t = ctrl_pts_law.dot(middle_point - astar_path[Astar_id]) / denom;
                        intersection_point = astar_path[Astar_id] + 
                            t * (astar_path[Astar_id] - astar_path[last_Astar_id]);
                    } else {
                        intersection_point = astar_path[Astar_id];
                    }
                    
                    if ((intersection_point - middle_point).norm() > 0.01) {
                        cps_.flag_temp[seg.first] = true;
                        cps_.base_point[seg.first].push_back(cps_.points.col(seg.first));
                        cps_.direction[seg.first].push_back(
                            (intersection_point - middle_point).normalized());
                        got_intersection_id = seg.first;
                    }
                    break;
                }
                
                last_val = val;
            }
        }
        
        // Step 3: Propagate directions to neighboring control points
        // CRITICAL: Original ego-planner uses final_segment_ids (which equals seg unless too short)
        // NOT bounds! This ensures only collision area control points get directions.
        if (got_intersection_id >= 0) {
            // Forward propagation - only within collision segment, not full bounds
            for (int j = got_intersection_id + 1; j <= seg.second && j < cps_.size; ++j) {
                if (!cps_.flag_temp[j] && !cps_.base_point[j - 1].empty()) {
                    cps_.base_point[j].push_back(cps_.base_point[j - 1].back());
                    cps_.direction[j].push_back(cps_.direction[j - 1].back());
                }
            }
            
            // Backward propagation - only within collision segment
            for (int j = got_intersection_id - 1; j >= seg.first && j >= 0; --j) {
                if (!cps_.flag_temp[j] && !cps_.base_point[j + 1].empty()) {
                    cps_.base_point[j].push_back(cps_.base_point[j + 1].back());
                    cps_.direction[j].push_back(cps_.direction[j + 1].back());
                }
            }
        }
    }
    
    void assignDirectionsFromAStar(const std::pair<int, int>& seg,
                                    const std::vector<Eigen::Vector3d>& astar_path) {
        // Use original-style implementation with default bounds
        std::pair<int, int> bound(seg.first, seg.second);
        assignDirectionsFromAStarOriginal(seg, bound, astar_path);
    }
    
    bool reboundOptimize(int max_iterations) {
        iter_num_ = 0;
        int start_id = order_;
        int end_id = cps_.size - order_;
        int variable_num = 3 * (end_id - start_id);
        
        if (variable_num <= 0) return false;
        
        new_lambda2_ = lambda_collision_;
        int restart_count = 0;
        constexpr int MAX_RESTART_NUMS = 5;  // Increased for better convergence
        
        while (restart_count < MAX_RESTART_NUMS) {
            force_stop_type_ = DONT_STOP;
            iter_num_ = 0;
            
            std::vector<double> x(variable_num);
            for (int i = start_id; i < end_id; ++i) {
                for (int j = 0; j < 3; ++j) {
                    x[3 * (i - start_id) + j] = cps_.points(j, i);
                }
            }
            
            lbfgs::lbfgs_parameter_t params;
            lbfgs::lbfgs_load_default_parameters(&params);
            params.mem_size = 16;
            params.max_iterations = max_iterations;
            params.g_epsilon = 0.01;
            
            double final_cost;
            int result = lbfgs::lbfgs_optimize(
                variable_num, x.data(), &final_cost,
                &BsplineOptimizer::costFunctionRebound,
                nullptr, &BsplineOptimizer::earlyExit, this, &params
            );
            
            for (int i = start_id; i < end_id; ++i) {
                for (int j = 0; j < 3; ++j) {
                    cps_.points(j, i) = x[3 * (i - start_id) + j];
                }
            }
            
            // Check collision on trajectory (2/3 like original)
            bool has_collision = false;
            UniformBspline temp_traj(cps_.points, order_, bspline_interval_);
            double duration = temp_traj.getDuration();
            double t_step = bspline_interval_ / 2;
            
            for (double t = 0; t < duration * 2.0 / 3.0; t += t_step) {
                if (isCollision(temp_traj.evaluateDeBoorT(t))) {
                    has_collision = true;
                    break;
                }
            }
            
            if (!has_collision) {
                return true;  // Success!
            }
            
            // Restart with increased collision weight
            restart_count++;
            initControlPointsWithAStar();
            new_lambda2_ *= 2.0;
        }
        
        return false;
    }
    
    static double costFunctionRebound(void* instance, const double* x, double* grad, int n) {
        BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(instance);
        double cost;
        opt->combineCostRebound(x, grad, cost, n);
        opt->iter_num_++;
        return cost;
    }
    
    /**
     * @brief Early exit callback for L-BFGS (same as original ego-planner)
     * 
     * This allows optimization to stop early when collision rebound is detected
     * or an error occurs, saving computation time.
     */
    static int earlyExit(void* instance, const double* x, const double* g, 
                         const double fx, const double xnorm, const double gnorm,
                         const double step, int n, int k, int ls) {
        BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(instance);
        return (opt->force_stop_type_ == STOP_FOR_ERROR || 
                opt->force_stop_type_ == STOP_FOR_REBOUND);
    }
    
    void combineCostRebound(const double* x, double* grad, double& f_combine, int n) {
        int start_id = order_;
        int end_id = cps_.size - order_;
        
        for (int i = start_id; i < end_id; ++i) {
            for (int j = 0; j < 3; ++j) {
                cps_.points(j, i) = x[3 * (i - start_id) + j];
            }
        }
        
        double f_smooth = 0, f_collision = 0, f_feasibility = 0;
        Eigen::MatrixXd g_smooth = Eigen::MatrixXd::Zero(3, cps_.size);
        Eigen::MatrixXd g_collision = Eigen::MatrixXd::Zero(3, cps_.size);
        Eigen::MatrixXd g_feasibility = Eigen::MatrixXd::Zero(3, cps_.size);
        
        calcSmoothnessCost(f_smooth, g_smooth);
        calcDistanceCostRebound(f_collision, g_collision, f_smooth);  // Pass smoothness for rebound check
        calcFeasibilityCost(f_feasibility, g_feasibility);
        
        f_combine = lambda_smooth_ * f_smooth + 
                    new_lambda2_ * f_collision + 
                    lambda_feasibility_ * f_feasibility;
        
        Eigen::MatrixXd grad_3d = lambda_smooth_ * g_smooth +
                                   new_lambda2_ * g_collision +
                                   lambda_feasibility_ * g_feasibility;
        
        for (int i = start_id; i < end_id; ++i) {
            for (int j = 0; j < 3; ++j) {
                grad[3 * (i - start_id) + j] = grad_3d(j, i);
            }
        }
    }
    
    // ==================== Refine Optimization (Stage 2) ====================
    
    bool refineOptimize(int max_iterations) {
        if (ref_pts_.empty()) return true;
        
        iter_num_ = 0;
        int start_id = order_;
        int end_id = cps_.size - order_;
        int variable_num = 3 * (end_id - start_id);
        
        if (variable_num <= 0) return false;
        
        std::vector<double> x(variable_num);
        for (int i = start_id; i < end_id; ++i) {
            for (int j = 0; j < 3; ++j) {
                x[3 * (i - start_id) + j] = cps_.points(j, i);
            }
        }
        
        lbfgs::lbfgs_parameter_t params;
        lbfgs::lbfgs_load_default_parameters(&params);
        params.mem_size = 16;
        params.max_iterations = max_iterations;
        params.g_epsilon = 0.001;
        
        double final_cost;
        lbfgs::lbfgs_optimize(
            variable_num, x.data(), &final_cost,
            &BsplineOptimizer::costFunctionRefine,
            nullptr, nullptr, this, &params
        );
        
        for (int i = start_id; i < end_id; ++i) {
            for (int j = 0; j < 3; ++j) {
                cps_.points(j, i) = x[3 * (i - start_id) + j];
            }
        }
        
        return true;
    }
    
    static double costFunctionRefine(void* instance, const double* x, double* grad, int n) {
        BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(instance);
        double cost;
        opt->combineCostRefine(x, grad, cost, n);
        opt->iter_num_++;
        return cost;
    }
    
    void combineCostRefine(const double* x, double* grad, double& f_combine, int n) {
        int start_id = order_;
        int end_id = cps_.size - order_;
        
        for (int i = start_id; i < end_id; ++i) {
            for (int j = 0; j < 3; ++j) {
                cps_.points(j, i) = x[3 * (i - start_id) + j];
            }
        }
        
        double f_smooth = 0, f_fitness = 0, f_feasibility = 0;
        Eigen::MatrixXd g_smooth = Eigen::MatrixXd::Zero(3, cps_.size);
        Eigen::MatrixXd g_fitness = Eigen::MatrixXd::Zero(3, cps_.size);
        Eigen::MatrixXd g_feasibility = Eigen::MatrixXd::Zero(3, cps_.size);
        
        calcSmoothnessCost(f_smooth, g_smooth);
        calcFitnessCost(f_fitness, g_fitness);
        calcFeasibilityCost(f_feasibility, g_feasibility);
        
        f_combine = lambda_smooth_ * f_smooth + 
                    lambda_fitness_ * f_fitness + 
                    lambda_feasibility_ * f_feasibility;
        
        Eigen::MatrixXd grad_3d = lambda_smooth_ * g_smooth +
                                   lambda_fitness_ * g_fitness +
                                   lambda_feasibility_ * g_feasibility;
        
        for (int i = start_id; i < end_id; ++i) {
            for (int j = 0; j < 3; ++j) {
                grad[3 * (i - start_id) + j] = grad_3d(j, i);
            }
        }
    }
    
    // ==================== Cost Functions ====================
    
    /**
     * @brief Smoothness cost (jerk minimization)
     */
    void calcSmoothnessCost(double& cost, Eigen::MatrixXd& gradient) {
        cost = 0.0;
        for (int i = 0; i < cps_.size - 3; ++i) {
            Eigen::Vector3d jerk = cps_.points.col(i + 3) - 3 * cps_.points.col(i + 2) +
                                   3 * cps_.points.col(i + 1) - cps_.points.col(i);
            cost += jerk.squaredNorm();
            
            Eigen::Vector3d temp_j = 2.0 * jerk;
            gradient.col(i + 0) += -temp_j;
            gradient.col(i + 1) += 3.0 * temp_j;
            gradient.col(i + 2) += -3.0 * temp_j;
            gradient.col(i + 3) += temp_j;
        }
    }
    
    /**
     * @brief Distance cost for collision avoidance (ego-planner style)
     * Uses base_point + direction from A* path guidance
     */
    void calcDistanceCostRebound(double& cost, Eigen::MatrixXd& gradient, double smoothness_cost) {
        cost = 0.0;
        int end_idx = cps_.size - order_;
        double demarcation = cps_.clearance;
        double a = 3 * demarcation;
        double b = -3 * demarcation * demarcation;
        double c = demarcation * demarcation * demarcation;
        
        force_stop_type_ = DONT_STOP;
        
        // Check for new collisions and rebound (same as original ego-planner)
        // 0.1 is an experimental value indicating trajectory is smooth enough
        if (iter_num_ > 3 && smoothness_cost / std::max(1, cps_.size - 2 * order_) < 0.1) {
            checkCollisionAndRebound();
        }
        
        for (int i = order_; i < end_idx; ++i) {
            for (size_t j = 0; j < cps_.direction[i].size(); ++j) {
                // Distance along push direction from base point
                double dist = (cps_.points.col(i) - cps_.base_point[i][j]).dot(cps_.direction[i][j]);
                double dist_err = cps_.clearance - dist;
                
                if (dist_err > 0) {
                    Eigen::Vector3d dist_grad = cps_.direction[i][j];
                    
                    if (dist_err < demarcation) {
                        cost += std::pow(dist_err, 3);
                        gradient.col(i) += -3.0 * dist_err * dist_err * dist_grad;
                    } else {
                        cost += a * dist_err * dist_err + b * dist_err + c;
                        gradient.col(i) += -(2.0 * a * dist_err + b) * dist_grad;
                    }
                }
            }
        }
    }
    
    /**
     * @brief Check for new collisions during optimization (same as original ego-planner)
     * 
     * This function detects new collisions that appear during optimization
     * and re-assigns directions using A* path guidance.
     */
    bool checkCollisionAndRebound() {
        int end_idx = cps_.size - order_;
        int i_end = end_idx - (end_idx - order_) / 3;
        
        double resolution = grid_map_ && grid_map_initialized_ ? grid_map_->getResolution() : 0.01;
        
        std::vector<std::pair<int, int>> segment_ids;
        bool flag_new_obs_valid = false;
        int in_id, out_id;
        
        for (int i = order_ - 1; i <= i_end; ++i) {
            bool occ = isCollision(cps_.points.col(i));
            
            // Check if the new collision will be valid (same as original)
            // Point is considered handled if it's already being pushed towards safety
            if (occ) {
                for (size_t k = 0; k < cps_.direction[i].size(); ++k) {
                    // Use grid resolution as threshold (same as original)
                    double dist = (cps_.points.col(i) - cps_.base_point[i][k]).dot(cps_.direction[i][k]);
                    if (dist < resolution) {
                        occ = false;
                        break;
                    }
                }
            }
            
            if (occ) {
                flag_new_obs_valid = true;
                
                // Find segment start (search backwards for free point)
                int j;
                for (j = i - 1; j >= 0; --j) {
                    if (!isCollision(cps_.points.col(j))) {
                        in_id = j;
                        break;
                    }
                }
                if (j < 0) {
                    // Failed to find free point - start is in obstacle
                    in_id = 0;
                }
                
                // Find segment end (search forwards for free point)
                for (j = i + 1; j < cps_.size; ++j) {
                    if (!isCollision(cps_.points.col(j))) {
                        out_id = j;
                        break;
                    }
                }
                if (j >= cps_.size) {
                    // Terminal point is in obstacle - error condition
                    force_stop_type_ = STOP_FOR_ERROR;
                    return false;
                }
                
                i = j + 1;  // Skip to after the segment
                segment_ids.push_back(std::make_pair(in_id, out_id));
            }
        }
        
        if (flag_new_obs_valid) {
            // Run A* for new segments and assign directions (same as original)
            std::vector<std::vector<Eigen::Vector3d>> a_star_paths;
            for (size_t i = 0; i < segment_ids.size(); ++i) {
                Eigen::Vector3d in_pt = cps_.points.col(segment_ids[i].first);
                Eigen::Vector3d out_pt = cps_.points.col(segment_ids[i].second);
                
                if (astar_->search(in_pt, out_pt)) {
                    a_star_paths.push_back(astar_->getPath());
                } else {
                    // A* failed, remove this segment
                    segment_ids.erase(segment_ids.begin() + i);
                    --i;
                    a_star_paths.push_back(std::vector<Eigen::Vector3d>());
                }
            }
            
            // Assign parameters to each segment (same as original)
            for (size_t i = 0; i < segment_ids.size(); ++i) {
                if (i < a_star_paths.size() && !a_star_paths[i].empty()) {
                    // Step 1: Clear temp flags
                    for (int j = segment_ids[i].first; j <= segment_ids[i].second && j < cps_.size; ++j) {
                        cps_.flag_temp[j] = false;
                    }
                    
                    // Step 2: Find intersection and assign directions
                    int got_intersection_id = -1;
                    for (int j = segment_ids[i].first + 1; j < segment_ids[i].second && j + 1 < cps_.size; ++j) {
                        Eigen::Vector3d ctrl_pts_law = cps_.points.col(j + 1) - cps_.points.col(j - 1);
                        Eigen::Vector3d intersection_point;
                        
                        int Astar_id = a_star_paths[i].size() / 2;
                        int last_Astar_id;
                        double val = (a_star_paths[i][Astar_id] - cps_.points.col(j)).dot(ctrl_pts_law);
                        double last_val = val;
                        
                        while (Astar_id >= 0 && Astar_id < (int)a_star_paths[i].size()) {
                            last_Astar_id = Astar_id;
                            
                            if (val >= 0)
                                --Astar_id;
                            else
                                ++Astar_id;
                            
                            if (Astar_id < 0 || Astar_id >= (int)a_star_paths[i].size()) break;
                            
                            val = (a_star_paths[i][Astar_id] - cps_.points.col(j)).dot(ctrl_pts_law);
                            
                            if (val * last_val <= 0 && (std::abs(val) > 0 || std::abs(last_val) > 0)) {
                                double denom = ctrl_pts_law.dot(a_star_paths[i][Astar_id] - a_star_paths[i][last_Astar_id]);
                                if (std::abs(denom) > 1e-10) {
                                    double t = ctrl_pts_law.dot(cps_.points.col(j) - a_star_paths[i][Astar_id]) / denom;
                                    intersection_point = a_star_paths[i][Astar_id] + 
                                        t * (a_star_paths[i][Astar_id] - a_star_paths[i][last_Astar_id]);
                                } else {
                                    intersection_point = a_star_paths[i][Astar_id];
                                }
                                
                                got_intersection_id = j;
                                break;
                            }
                            
                            last_val = val;
                        }
                        
                        if (got_intersection_id >= 0) {
                            cps_.flag_temp[j] = true;
                            double length = (intersection_point - cps_.points.col(j)).norm();
                            
                            if (length > 1e-5) {
                                for (double a = length; a >= 0.0; a -= resolution) {
                                    Eigen::Vector3d test_pt = (a / length) * intersection_point + 
                                                              (1 - a / length) * cps_.points.col(j);
                                    bool test_occ = isCollision(test_pt);
                                    
                                    if (test_occ || a < resolution) {
                                        if (test_occ) a += resolution;
                                        cps_.base_point[j].push_back(
                                            (a / length) * intersection_point + 
                                            (1 - a / length) * cps_.points.col(j));
                                        cps_.direction[j].push_back(
                                            (intersection_point - cps_.points.col(j)).normalized());
                                        break;
                                    }
                                }
                            } else {
                                got_intersection_id = -1;
                            }
                        }
                    }
                    
                    // Step 3: Propagate directions
                    if (got_intersection_id >= 0) {
                        for (int j = got_intersection_id + 1; j <= segment_ids[i].second && j < cps_.size; ++j) {
                            if (!cps_.flag_temp[j] && !cps_.base_point[j - 1].empty()) {
                                cps_.base_point[j].push_back(cps_.base_point[j - 1].back());
                                cps_.direction[j].push_back(cps_.direction[j - 1].back());
                            }
                        }
                        
                        for (int j = got_intersection_id - 1; j >= segment_ids[i].first && j >= 0; --j) {
                            if (!cps_.flag_temp[j] && !cps_.base_point[j + 1].empty()) {
                                cps_.base_point[j].push_back(cps_.base_point[j + 1].back());
                                cps_.direction[j].push_back(cps_.direction[j + 1].back());
                            }
                        }
                    }
                }
            }
            
            force_stop_type_ = STOP_FOR_REBOUND;
            return true;
        }
        
        return false;
    }
    
    /**
     * @brief Feasibility cost (velocity and acceleration limits)
     * Exactly matches original ego-planner's non-SECOND_DERIVATIVE_CONTINOUS version
     */
    void calcFeasibilityCost(double& cost, Eigen::MatrixXd& gradient) {
        cost = 0.0;
        
        double ts = bspline_interval_;
        double ts_inv2 = 1.0 / ts / ts;
        
        // Velocity feasibility (same as original ego-planner)
        for (int i = 0; i < cps_.size - 1; ++i) {
            Eigen::Vector3d vi = (cps_.points.col(i + 1) - cps_.points.col(i)) / ts;
            
            for (int j = 0; j < 3; ++j) {
                if (vi(j) > max_vel_) {
                    // Cost: (vi - max_vel)^2 * ts_inv2
                    cost += std::pow(vi(j) - max_vel_, 2) * ts_inv2;
                    
                    // Gradient: exactly as original
                    gradient(j, i + 0) += -2 * (vi(j) - max_vel_) / ts * ts_inv2;
                    gradient(j, i + 1) += 2 * (vi(j) - max_vel_) / ts * ts_inv2;
                } else if (vi(j) < -max_vel_) {
                    cost += std::pow(vi(j) + max_vel_, 2) * ts_inv2;
                    
                    gradient(j, i + 0) += -2 * (vi(j) + max_vel_) / ts * ts_inv2;
                    gradient(j, i + 1) += 2 * (vi(j) + max_vel_) / ts * ts_inv2;
                }
            }
        }
        
        // Acceleration feasibility (same as original ego-planner)
        for (int i = 0; i < cps_.size - 2; ++i) {
            Eigen::Vector3d ai = (cps_.points.col(i + 2) - 2 * cps_.points.col(i + 1) + 
                                  cps_.points.col(i)) * ts_inv2;
            
            for (int j = 0; j < 3; ++j) {
                if (ai(j) > max_acc_) {
                    // Cost: (ai - max_acc)^2
                    cost += std::pow(ai(j) - max_acc_, 2);
                    
                    // Gradient: exactly as original
                    gradient(j, i + 0) += 2 * (ai(j) - max_acc_) * ts_inv2;
                    gradient(j, i + 1) += -4 * (ai(j) - max_acc_) * ts_inv2;
                    gradient(j, i + 2) += 2 * (ai(j) - max_acc_) * ts_inv2;
                } else if (ai(j) < -max_acc_) {
                    cost += std::pow(ai(j) + max_acc_, 2);
                    
                    gradient(j, i + 0) += 2 * (ai(j) + max_acc_) * ts_inv2;
                    gradient(j, i + 1) += -4 * (ai(j) + max_acc_) * ts_inv2;
                    gradient(j, i + 2) += 2 * (ai(j) + max_acc_) * ts_inv2;
                }
            }
        }
    }
    
    /**
     * @brief Fitness cost for curve fitting (Stage 2)
     * 
     * f = |x·v|²/a² + |x×v|²/b²
     * where:
     * - x is deviation from reference point
     * - v is tangent direction of reference trajectory
     * - a² = 25 (low penalty for axial deviation)
     * - b² = 1  (high penalty for radial deviation)
     * 
     * This allows adjustment along the trajectory but penalizes deviation perpendicular to it
     */
    void calcFitnessCost(double& cost, Eigen::MatrixXd& gradient) {
        cost = 0.0;
        if (ref_pts_.size() < 3) return;
        
        int end_idx = cps_.size - order_;
        
        // Axial vs radial penalty ratio: radial is penalized 25x more
        double a2 = 25.0;  // Axial (along trajectory)
        double b2 = 1.0;   // Radial (perpendicular)
        
        for (int i = order_ - 1; i < end_idx + 1 && i + 1 < (int)ref_pts_.size(); ++i) {
            if (i < 1 || i >= cps_.size - 1) continue;
            
            // B-spline point approximation
            Eigen::Vector3d bspline_pt = (cps_.points.col(i - 1) + 4 * cps_.points.col(i) + 
                                          cps_.points.col(i + 1)) / 6.0;
            
            // Deviation from reference
            Eigen::Vector3d x = bspline_pt - ref_pts_[i];
            
            // Tangent direction of reference trajectory
            Eigen::Vector3d v = Eigen::Vector3d::Zero();
            if (i >= 1 && i + 1 < (int)ref_pts_.size()) {
                v = (ref_pts_[i + 1] - ref_pts_[i - 1]);
                double v_norm = v.norm();
                if (v_norm > 1e-6) v /= v_norm;
            }
            
            if (v.norm() < 1e-6) continue;
            
            double xdotv = x.dot(v);
            Eigen::Vector3d xcrossv = x.cross(v);
            
            // Cost: penalize radial deviation more than axial
            double f = (xdotv * xdotv) / a2 + xcrossv.squaredNorm() / b2;
            cost += f;
            
            // Gradient
            Eigen::Matrix3d m;
            m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
            Eigen::Vector3d df_dx = 2 * xdotv / a2 * v + 2 / b2 * m * xcrossv;
            
            gradient.col(i - 1) += df_dx / 6.0;
            gradient.col(i) += 4 * df_dx / 6.0;
            gradient.col(i + 1) += df_dx / 6.0;
        }
    }
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace ego_planner

#endif // BSPLINE_OPTIMIZER_HPP
