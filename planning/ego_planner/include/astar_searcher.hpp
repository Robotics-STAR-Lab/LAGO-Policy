/**
 * @file astar_searcher.hpp
 * @brief A* path searcher for AABB obstacle avoidance
 * 
 * Optimized implementation based on ego-planner's dyn_a_star
 * Key optimizations:
 * - Pre-computed occupancy grid for O(1) collision check
 * - Larger step size (0.1m instead of 0.01m)
 * - Simplified findNearestFree
 */

#ifndef ASTAR_SEARCHER_HPP
#define ASTAR_SEARCHER_HPP

#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <memory>
#include <cmath>
#include <limits>

namespace ego_planner {

/**
 * @brief Axis-Aligned Bounding Box for obstacle representation
 */
struct AABB {
    Eigen::Vector3d min_corner;
    Eigen::Vector3d max_corner;
    
    AABB() = default;
    AABB(const Eigen::Vector3d& min_c, const Eigen::Vector3d& max_c)
        : min_corner(min_c), max_corner(max_c) {}
};

/**
 * @brief Grid node for A* search
 */
struct GridNode {
    enum State { UNDEFINED, OPENSET, CLOSEDSET };
    
    Eigen::Vector3i index;
    State state = UNDEFINED;
    double g_score = std::numeric_limits<double>::infinity();
    double f_score = std::numeric_limits<double>::infinity();
    GridNode* came_from = nullptr;
    int round = 0;
};

/**
 * @brief Comparator for priority queue
 */
struct NodeComparator {
    bool operator()(GridNode* a, GridNode* b) {
        return a->f_score > b->f_score;
    }
};

/**
 * @brief A* path searcher with pre-computed occupancy grid (O(1) collision check)
 */
class AStarSearcher {
public:
    AStarSearcher() = default;
    
    /**
     * @brief Initialize the grid map
     * @param resolution Grid resolution (m) - use larger values (0.05-0.1) for speed
     * @param origin Grid origin (minimum corner)
     * @param size Grid size in each dimension
     */
    void init(double resolution, const Eigen::Vector3d& origin, const Eigen::Vector3i& size) {
        resolution_ = resolution;
        inv_resolution_ = 1.0 / resolution;
        origin_ = origin;
        grid_size_ = size;
        
        // Pre-allocate grid nodes (3D array for cache efficiency)
        int total = size(0) * size(1) * size(2);
        nodes_.resize(total);
        for (int i = 0; i < total; ++i) {
            nodes_[i].state = GridNode::UNDEFINED;
            nodes_[i].round = 0;
        }
        
        // Pre-allocate occupancy grid for O(1) collision check
        occupancy_.assign(total, false);
    }
    
    /**
     * @brief Set obstacles and pre-compute occupancy grid
     */
    void setObstacles(const std::vector<AABB>& obstacles, double safety_margin = 0.0) {
        obstacles_ = &obstacles;
        safety_margin_ = safety_margin;
        
        // Clear and rebuild occupancy grid
        std::fill(occupancy_.begin(), occupancy_.end(), false);
        
        // Fill occupancy grid from AABB obstacles
        for (const auto& obs : obstacles) {
            Eigen::Vector3d min_pt = obs.min_corner - Eigen::Vector3d::Constant(safety_margin);
            Eigen::Vector3d max_pt = obs.max_corner + Eigen::Vector3d::Constant(safety_margin);
            
            Eigen::Vector3i min_idx, max_idx;
            posToIndexClamped(min_pt, min_idx);
            posToIndexClamped(max_pt, max_idx);
            
            // Fill cells
            for (int x = min_idx(0); x <= max_idx(0); ++x) {
                for (int y = min_idx(1); y <= max_idx(1); ++y) {
                    for (int z = min_idx(2); z <= max_idx(2); ++z) {
                        if (isValidIndex(Eigen::Vector3i(x, y, z))) {
                            occupancy_[indexToFlat(Eigen::Vector3i(x, y, z))] = true;
                        }
                    }
                }
            }
        }
    }
    
    /**
     * @brief Search for a collision-free path (optimized like original ego-planner)
     */
    bool search(const Eigen::Vector3d& start, const Eigen::Vector3d& goal) {
        ++current_round_;
        path_.clear();
        
        // Set step size and center (like original)
        step_size_ = resolution_;
        center_ = 0.5 * (start + goal);
        
        Eigen::Vector3i start_idx, goal_idx;
        if (!convertAndAdjustPoints(start, goal, start_idx, goal_idx)) {
            return false;
        }
        
        // Get start and goal nodes
        GridNode* start_node = &nodes_[indexToFlat(start_idx)];
        GridNode* goal_node = &nodes_[indexToFlat(goal_idx)];
        
        // Clear open set
        while (!open_set_.empty()) open_set_.pop();
        
        // Initialize start node
        start_node->index = start_idx;
        start_node->round = current_round_;
        start_node->g_score = 0;
        start_node->f_score = getHeuristic(start_idx, goal_idx);
        start_node->state = GridNode::OPENSET;
        start_node->came_from = nullptr;
        open_set_.push(start_node);
        
        goal_node->index = goal_idx;
        
        int num_iter = 0;
        const int max_iter = 10000;  // Limit iterations for speed
        
        while (!open_set_.empty() && num_iter++ < max_iter) {
            GridNode* current = open_set_.top();
            open_set_.pop();
            
            // Check if reached goal
            if (current->index == goal_idx) {
                // Reconstruct path
                GridNode* node = current;
                while (node != nullptr) {
                    path_.push_back(indexToPos(node->index));
                    node = node->came_from;
                }
                std::reverse(path_.begin(), path_.end());
                return true;
            }
            
            current->state = GridNode::CLOSEDSET;
            
            // Expand 26-connected neighbors
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;
                        
                        Eigen::Vector3i neighbor_idx(
                            current->index(0) + dx,
                            current->index(1) + dy,
                            current->index(2) + dz
                        );
                        
                        // Bounds check (with margin like original)
                        if (neighbor_idx(0) < 1 || neighbor_idx(0) >= grid_size_(0) - 1 ||
                            neighbor_idx(1) < 1 || neighbor_idx(1) >= grid_size_(1) - 1 ||
                            neighbor_idx(2) < 1 || neighbor_idx(2) >= grid_size_(2) - 1) {
                            continue;
                        }
                        
                        GridNode* neighbor = &nodes_[indexToFlat(neighbor_idx)];
                        
                        // Check if already explored in this round
                        bool flag_explored = (neighbor->round == current_round_);
                        
                        // Skip if in closed set
                        if (flag_explored && neighbor->state == GridNode::CLOSEDSET) {
                            continue;
                        }
                        
                        neighbor->round = current_round_;
                        
                        // O(1) collision check using pre-computed occupancy grid!
                        if (occupancy_[indexToFlat(neighbor_idx)]) {
                            continue;
                        }
                        
                        // Compute cost with STRONG vertical preference
                        double base_cost = std::sqrt(dx*dx + dy*dy + dz*dz);
                        double move_cost = base_cost;
                        
                        if (prefer_vertical_) {
                            // Strong preference for vertical movement (Z-axis)
                            // - Pure vertical move (dx=0, dy=0, dz!=0): heavy discount
                            // - Horizontal move (dz=0): heavy penalty
                            // - Mixed move: proportional
                            double horiz_component = dx*dx + dy*dy;
                            double vert_component = dz*dz;
                            
                            if (horiz_component < 0.01 && vert_component > 0.01) {
                                // Pure vertical: apply strong discount
                                move_cost = base_cost * vertical_discount_;
                            } else if (horiz_component > 0.01 && vert_component < 0.01) {
                                // Pure horizontal: apply penalty (make it more expensive)
                                move_cost = base_cost * (2.0 - vertical_discount_);
                            } else if (vert_component > 0.01) {
                                // Mixed with vertical: small discount based on Z ratio
                                double z_ratio = std::abs(dz) / base_cost;
                                move_cost = base_cost * (1.0 - z_ratio * (1.0 - vertical_discount_));
                            }
                        }
                        
                        double tentative_g = current->g_score + move_cost;
                        
                        if (!flag_explored) {
                            // Discover new node
                            neighbor->index = neighbor_idx;
                            neighbor->state = GridNode::OPENSET;
                            neighbor->came_from = current;
                            neighbor->g_score = tentative_g;
                            neighbor->f_score = tentative_g + getHeuristic(neighbor_idx, goal_idx);
                            open_set_.push(neighbor);
                        } else if (tentative_g < neighbor->g_score) {
                            // Update existing node
                            neighbor->came_from = current;
                            neighbor->g_score = tentative_g;
                            neighbor->f_score = tentative_g + getHeuristic(neighbor_idx, goal_idx);
                        }
                    }
                }
            }
        }
        
        return false;
    }
    
    /**
     * @brief Get the found path
     */
    const std::vector<Eigen::Vector3d>& getPath() const { return path_; }
    
public:
    /**
     * @brief Set preference for vertical (Z-axis) movement
     * When enabled, A* will prefer paths that go up/down rather than sideways.
     * This is useful for robotic arm obstacle avoidance where going over is preferred.
     * 
     * @param prefer Enable vertical preference
     * @param discount Cost discount for vertical movement (0.5 = 50% discount, default 0.3 = 70% discount)
     */
    void setPreferVertical(bool prefer, double discount = 0.3) {
        prefer_vertical_ = prefer;
        vertical_discount_ = discount;
    }
    
    bool getPreferVertical() const { return prefer_vertical_; }

private:
    double resolution_ = 0.1;  // Larger default for speed
    double inv_resolution_ = 10.0;
    double step_size_ = 0.1;
    Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
    Eigen::Vector3i grid_size_ = Eigen::Vector3i(100, 100, 100);
    double safety_margin_ = 0.0;
    bool prefer_vertical_ = true;  // Default: prefer vertical movement for robotic arm
    double vertical_discount_ = 0.3;  // 70% discount for Z movement (cost * 0.3)
    
    std::vector<GridNode> nodes_;  // Flat array (more cache-friendly than unique_ptr)
    std::vector<bool> occupancy_;  // Pre-computed occupancy for O(1) check
    const std::vector<AABB>* obstacles_ = nullptr;
    std::vector<Eigen::Vector3d> path_;
    std::priority_queue<GridNode*, std::vector<GridNode*>, NodeComparator> open_set_;
    int current_round_ = 0;
    
    bool posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const {
        Eigen::Vector3d diff = (pos - origin_) * inv_resolution_;
        idx = Eigen::Vector3i(
            (int)std::floor(diff(0)),
            (int)std::floor(diff(1)),
            (int)std::floor(diff(2))
        );
        return isValidIndex(idx);
    }
    
    void posToIndexClamped(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const {
        Eigen::Vector3d diff = (pos - origin_) * inv_resolution_;
        idx = Eigen::Vector3i(
            std::max(0, std::min(grid_size_(0) - 1, (int)std::floor(diff(0)))),
            std::max(0, std::min(grid_size_(1) - 1, (int)std::floor(diff(1)))),
            std::max(0, std::min(grid_size_(2) - 1, (int)std::floor(diff(2))))
        );
    }
    
    Eigen::Vector3d indexToPos(const Eigen::Vector3i& idx) const {
        return origin_ + (idx.cast<double>() + Eigen::Vector3d(0.5, 0.5, 0.5)) * resolution_;
    }
    
    bool isValidIndex(const Eigen::Vector3i& idx) const {
        return idx(0) >= 0 && idx(0) < grid_size_(0) &&
               idx(1) >= 0 && idx(1) < grid_size_(1) &&
               idx(2) >= 0 && idx(2) < grid_size_(2);
    }
    
    int indexToFlat(const Eigen::Vector3i& idx) const {
        return idx(0) + grid_size_(0) * (idx(1) + grid_size_(1) * idx(2));
    }
    
    /**
     * @brief Diagonal distance heuristic (same as original ego-planner)
     */
    double getHeuristic(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const {
        double dx = std::abs(a(0) - b(0));
        double dy = std::abs(a(1) - b(1));
        double dz = std::abs(a(2) - b(2));
        
        double diag = std::min({dx, dy, dz});
        dx -= diag;
        dy -= diag;
        dz -= diag;
        
        double h = 0.0;
        if (dx == 0) {
            h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dy, dz) + std::abs(dy - dz);
        } else if (dy == 0) {
            h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dz) + std::abs(dx - dz);
        } else {
            h = std::sqrt(3.0) * diag + std::sqrt(2.0) * std::min(dx, dy) + std::abs(dx - dy);
        }
        
        // Tie breaker (same as original)
        return h * 1.0001;
    }
    
    /**
     * @brief Convert coordinates and adjust start/end if in obstacle (like original)
     */
    bool convertAndAdjustPoints(const Eigen::Vector3d& start_pt, const Eigen::Vector3d& end_pt,
                                 Eigen::Vector3i& start_idx, Eigen::Vector3i& end_idx) {
        if (!posToIndex(start_pt, start_idx) || !posToIndex(end_pt, end_idx)) {
            return false;
        }
        
        // Adjust start if in obstacle (simple linear search like original)
        Eigen::Vector3d adjusted_start = start_pt;
        Eigen::Vector3d direction = (end_pt - start_pt).normalized();
        int max_adjust = 50;
        
        while (isValidIndex(start_idx) && occupancy_[indexToFlat(start_idx)] && max_adjust-- > 0) {
            adjusted_start = adjusted_start - direction * step_size_;
            if (!posToIndex(adjusted_start, start_idx)) {
                // Try going up instead
                adjusted_start = start_pt + Eigen::Vector3d(0, 0, step_size_);
                if (!posToIndex(adjusted_start, start_idx)) return false;
            }
        }
        
        // Adjust end if in obstacle
        Eigen::Vector3d adjusted_end = end_pt;
        direction = (start_pt - end_pt).normalized();
        max_adjust = 50;
        
        while (isValidIndex(end_idx) && occupancy_[indexToFlat(end_idx)] && max_adjust-- > 0) {
            adjusted_end = adjusted_end - direction * step_size_;
            if (!posToIndex(adjusted_end, end_idx)) {
                // Try going up instead
                adjusted_end = end_pt + Eigen::Vector3d(0, 0, step_size_);
                if (!posToIndex(adjusted_end, end_idx)) return false;
            }
        }
        
        return isValidIndex(start_idx) && isValidIndex(end_idx);
    }
};

} // namespace ego_planner

#endif // ASTAR_SEARCHER_HPP
