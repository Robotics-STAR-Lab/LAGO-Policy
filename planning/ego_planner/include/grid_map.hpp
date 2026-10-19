/**
 * @file grid_map.hpp
 * @brief 3D Occupancy Grid Map for O(1) collision detection
 * 
 * Similar to ego-planner's GridMap, but standalone without ROS.
 * Pre-computes occupancy from AABB obstacles for fast collision checking.
 */

#ifndef GRID_MAP_HPP
#define GRID_MAP_HPP

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace ego_planner {

struct AABB;  // Forward declaration

/**
 * @brief 3D Occupancy Grid Map
 * 
 * Provides O(1) collision detection by pre-computing occupancy grid.
 */
class GridMap {
public:
    GridMap() = default;
    
    /**
     * @brief Initialize grid map
     * @param resolution Grid resolution in meters
     * @param origin Origin point (min corner) of the grid
     * @param size Grid size in each dimension (number of cells)
     */
    void init(double resolution, const Eigen::Vector3d& origin, const Eigen::Vector3i& size) {
        resolution_ = resolution;
        inv_resolution_ = 1.0 / resolution;
        origin_ = origin;
        grid_size_ = size;
        
        // Allocate occupancy grid
        int total_cells = size(0) * size(1) * size(2);
        occupancy_.resize(total_cells, false);
        inflate_occupancy_.resize(total_cells, false);
        
        // Compute grid bounds
        map_min_ = origin;
        map_max_ = origin + size.cast<double>() * resolution;
    }
    
    /**
     * @brief Initialize grid map to cover given bounds with margin
     */
    void initFromBounds(double resolution, const Eigen::Vector3d& min_pt, 
                        const Eigen::Vector3d& max_pt, double margin = 0.5) {
        resolution_ = resolution;
        inv_resolution_ = 1.0 / resolution;
        
        // Expand bounds by margin
        origin_ = min_pt - Eigen::Vector3d::Constant(margin);
        Eigen::Vector3d max_expanded = max_pt + Eigen::Vector3d::Constant(margin);
        
        // Calculate grid size
        Eigen::Vector3d size_d = (max_expanded - origin_) * inv_resolution_;
        grid_size_ = Eigen::Vector3i(
            std::max(10, (int)std::ceil(size_d(0))),
            std::max(10, (int)std::ceil(size_d(1))),
            std::max(10, (int)std::ceil(size_d(2)))
        );
        
        // Limit size to prevent memory issues
        grid_size_(0) = std::min(grid_size_(0), 500);
        grid_size_(1) = std::min(grid_size_(1), 500);
        grid_size_(2) = std::min(grid_size_(2), 500);
        
        // Allocate occupancy grid
        int total_cells = grid_size_(0) * grid_size_(1) * grid_size_(2);
        occupancy_.resize(total_cells, false);
        inflate_occupancy_.resize(total_cells, false);
        
        // Compute grid bounds
        map_min_ = origin_;
        map_max_ = origin_ + grid_size_.cast<double>() * resolution_;
    }
    
    /**
     * @brief Set obstacles and compute occupancy grid
     * @param obstacles List of AABB obstacles
     * @param inflate_dist Inflation distance for safety margin
     */
    void setObstacles(const std::vector<AABB>& obstacles, double inflate_dist = 0.0) {
        obstacles_ = &obstacles;
        inflate_dist_ = inflate_dist;
        
        // Clear grids
        std::fill(occupancy_.begin(), occupancy_.end(), false);
        std::fill(inflate_occupancy_.begin(), inflate_occupancy_.end(), false);
        
        // Fill occupancy grid
        for (const auto& obs : obstacles) {
            fillOccupancy(obs, false);
            fillOccupancy(obs, true);
        }
    }
    
    /**
     * @brief Add a single obstacle and update grid
     */
    void addObstacle(const Eigen::Vector3d& min_corner, const Eigen::Vector3d& max_corner) {
        AABB obs{min_corner, max_corner};
        fillOccupancy(obs, false);
        fillOccupancy(obs, true);
    }
    
    /**
     * @brief Clear all obstacles
     */
    void clear() {
        std::fill(occupancy_.begin(), occupancy_.end(), false);
        std::fill(inflate_occupancy_.begin(), inflate_occupancy_.end(), false);
    }
    
    /**
     * @brief Get occupancy at a position (O(1))
     */
    bool getOccupancy(const Eigen::Vector3d& pos) const {
        Eigen::Vector3i idx;
        if (!posToIndex(pos, idx)) return false;  // Out of bounds = free
        return occupancy_[indexToFlat(idx)];
    }
    
    /**
     * @brief Get inflated occupancy at a position (O(1))
     * This is the main function for collision checking during optimization
     */
    bool getInflateOccupancy(const Eigen::Vector3d& pos) const {
        Eigen::Vector3i idx;
        if (!posToIndex(pos, idx)) {
            // Out of bounds - fallback to AABB check for safety
            if (obstacles_) {
                for (const auto& obs : *obstacles_) {
                    if (signedDistanceToAABB(pos, obs) < inflate_dist_) {
                        return true;
                    }
                }
            }
            return false;
        }
        return inflate_occupancy_[indexToFlat(idx)];
    }
    
    /**
     * @brief Check if a position is in collision (O(1))
     */
    bool isCollision(const Eigen::Vector3d& pos) const {
        return getInflateOccupancy(pos);
    }
    
    /**
     * @brief Get signed distance to nearest obstacle (approximate)
     * Positive = outside, Negative = inside
     */
    double getDistance(const Eigen::Vector3d& pos) const {
        if (!obstacles_) return 1.0;
        
        double min_dist = std::numeric_limits<double>::max();
        for (const auto& obs : *obstacles_) {
            double dist = signedDistanceToAABB(pos, obs);
            if (dist < min_dist) min_dist = dist;
        }
        return min_dist;
    }
    
    /**
     * @brief Get distance gradient (points away from obstacles)
     */
    Eigen::Vector3d getDistanceGradient(const Eigen::Vector3d& pos) const {
        if (!obstacles_) return Eigen::Vector3d::Zero();
        
        double min_dist = std::numeric_limits<double>::max();
        Eigen::Vector3d grad = Eigen::Vector3d::Zero();
        
        for (const auto& obs : *obstacles_) {
            double dist = signedDistanceToAABB(pos, obs);
            if (dist < min_dist) {
                min_dist = dist;
                grad = gradientDistanceToAABB(pos, obs);
            }
        }
        return grad;
    }
    
    // Getters
    double getResolution() const { return resolution_; }
    const Eigen::Vector3d& getOrigin() const { return origin_; }
    const Eigen::Vector3i& getGridSize() const { return grid_size_; }
    const Eigen::Vector3d& getMapMin() const { return map_min_; }
    const Eigen::Vector3d& getMapMax() const { return map_max_; }
    double getInflateDist() const { return inflate_dist_; }
    
private:
    double resolution_ = 0.01;
    double inv_resolution_ = 100.0;
    Eigen::Vector3d origin_ = Eigen::Vector3d::Zero();
    Eigen::Vector3i grid_size_ = Eigen::Vector3i(100, 100, 100);
    Eigen::Vector3d map_min_, map_max_;
    double inflate_dist_ = 0.0;
    
    std::vector<bool> occupancy_;
    std::vector<bool> inflate_occupancy_;
    const std::vector<AABB>* obstacles_ = nullptr;
    
    bool posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& idx) const {
        Eigen::Vector3d diff = (pos - origin_) * inv_resolution_;
        idx = Eigen::Vector3i(
            (int)std::floor(diff(0)),
            (int)std::floor(diff(1)),
            (int)std::floor(diff(2))
        );
        return isValidIndex(idx);
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
        // Match original ego-planner: x * Y * Z + y * Z + z
        // This ensures Z changes fastest, then Y, then X (row-major order)
        return idx(0) * grid_size_(1) * grid_size_(2) + idx(1) * grid_size_(2) + idx(2);
    }
    
    /**
     * @brief Fill occupancy grid for an AABB obstacle
     */
    void fillOccupancy(const AABB& obs, bool inflate) {
        double margin = inflate ? inflate_dist_ : 0.0;
        
        // Get index range for the obstacle
        Eigen::Vector3d min_pt = obs.min_corner - Eigen::Vector3d::Constant(margin);
        Eigen::Vector3d max_pt = obs.max_corner + Eigen::Vector3d::Constant(margin);
        
        Eigen::Vector3i min_idx, max_idx;
        posToIndex(min_pt, min_idx);
        posToIndex(max_pt, max_idx);
        
        // Clamp to valid range
        min_idx = min_idx.cwiseMax(Eigen::Vector3i::Zero());
        max_idx = max_idx.cwiseMin(grid_size_ - Eigen::Vector3i::Ones());
        
        // Fill cells
        auto& grid = inflate ? inflate_occupancy_ : occupancy_;
        for (int x = min_idx(0); x <= max_idx(0); ++x) {
            for (int y = min_idx(1); y <= max_idx(1); ++y) {
                for (int z = min_idx(2); z <= max_idx(2); ++z) {
                    grid[indexToFlat(Eigen::Vector3i(x, y, z))] = true;
                }
            }
        }
    }
    
    double signedDistanceToAABB(const Eigen::Vector3d& point, const AABB& box) const {
        Eigen::Vector3d center = 0.5 * (box.min_corner + box.max_corner);
        Eigen::Vector3d half = 0.5 * (box.max_corner - box.min_corner);
        Eigen::Vector3d d = (point - center).cwiseAbs() - half;
        double outside_dist = d.cwiseMax(0.0).norm();
        double inside_dist = std::min(d.maxCoeff(), 0.0);
        return outside_dist + inside_dist;
    }
    
    Eigen::Vector3d gradientDistanceToAABB(const Eigen::Vector3d& point, const AABB& box) const {
        Eigen::Vector3d center = 0.5 * (box.min_corner + box.max_corner);
        Eigen::Vector3d half = 0.5 * (box.max_corner - box.min_corner);
        Eigen::Vector3d d = point - center;
        
        Eigen::Vector3d grad = Eigen::Vector3d::Zero();
        for (int i = 0; i < 3; ++i) {
            if (d(i) > half(i)) {
                grad(i) = 1.0;
            } else if (d(i) < -half(i)) {
                grad(i) = -1.0;
            } else {
                // Inside this dimension - find closest face
                double d_pos = half(i) - d(i);
                double d_neg = half(i) + d(i);
                if (d_pos < d_neg) {
                    grad(i) = 1.0;
                } else {
                    grad(i) = -1.0;
                }
            }
        }
        
        double norm = grad.norm();
        if (norm > 1e-6) grad /= norm;
        return grad;
    }
};

}  // namespace ego_planner

#endif  // GRID_MAP_HPP

