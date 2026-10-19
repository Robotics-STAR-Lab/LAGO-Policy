"""
MINCO Trajectory Optimizer for Robot Arms (Cartesian Space)

This module provides MINCO trajectory optimization for end-effector trajectories:
- Position (x, y, z): MINCO polynomial optimization (minimum-snap)
- Rotation (rx, ry, rz): Slerp interpolation (quaternion-aware)

No retraining required - works as post-processing on diffusion policy outputs.

Features:
- Time-optimized Cartesian trajectory generation
- Position velocity/acceleration limit enforcement
- Smooth minimum-snap trajectory generation
- Configurable waypoint constraints (pass-through or relaxed)

Usage:
    optimizer = CartesianMincoOptimizer(cartesian_limits)
    success, traj, opt_times = optimizer.optimize(
        tcp_waypoints,    # (N, 6) waypoints [x, y, z, rx, ry, rz]
        init_times,       # (N-1,) initial time allocation
    )
    
    # Sample the trajectory
    if success:
        pose = traj.get_tcp_pose(t)  # Get pose at time t
"""

import sys
import os
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, List
from dataclasses import dataclass
import scipy.spatial.transform as st

# Path to the compiled MINCO library (``joint_traj_opt`` pybind module). MINCO is
# bundled in this repo under ``planning/minco/`` and built from source like
# ego-planner. This optimizer is OPTIONAL: it is only used for deployment-time
# 200Hz spatial-temporal trajectory smoothing. By default we look in the bundled
# build directory (``<repo_root>/planning/minco/build``); set MINCO_BUILD_DIR to
# override with a custom build location.
_DEFAULT_MINCO_BUILD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'minco', 'build',
)
MINCO_BUILD_DIR = os.environ.get('MINCO_BUILD_DIR') or _DEFAULT_MINCO_BUILD_DIR

if MINCO_BUILD_DIR and os.path.exists(MINCO_BUILD_DIR) and MINCO_BUILD_DIR not in sys.path:
    sys.path.insert(0, MINCO_BUILD_DIR)

try:
    import joint_traj_opt as jto
    MINCO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MINCO not available: {e}")
    print(f"  Expected at: {MINCO_BUILD_DIR}")
    print(f"  Set MINCO_BUILD_DIR environment variable to specify the correct path.")
    MINCO_AVAILABLE = False


@dataclass
class CartesianLimits:
    """Limits for Cartesian space motion."""
    pos_min: npt.NDArray[np.float64]  # (3,) min position [x, y, z]
    pos_max: npt.NDArray[np.float64]  # (3,) max position [x, y, z]
    vel_max: npt.NDArray[np.float64]  # (3,) max velocity [vx, vy, vz]
    acc_max: npt.NDArray[np.float64]  # (3,) max acceleration


# Default Cartesian limits for the robot
# Tuning guide for response speed:
#   - vel_max: Higher = faster motion (Franka max ~1.7 m/s)
#   - acc_max: Higher = faster acceleration (Franka max ~13 m/s²)
#   - wei_time in optimizer: Higher = more aggressive time optimization
DEFAULT_ROBOT_CARTESIAN_LIMITS = CartesianLimits(
    pos_min=np.array([-1.0, -1.0, -0.02]),
    pos_max=np.array([1.0, 1.0, 1.2]),
    vel_max=np.array([1.7, 1.7, 1.7]),  # Moderate limits (was 1.0)
    acc_max=np.array([13.0, 13.0, 13.0]),  # Moderate limits (was 5.0)
)


class CartesianMincoTrajectory:
    """
    Trajectory that combines:
    - MINCO-optimized position trajectory (x, y, z)
    - Slerp-interpolated rotation trajectory (rx, ry, rz as rotvec)
    
    This provides smooth position trajectories while maintaining proper
    rotation interpolation on SO(3).
    """
    
    def __init__(
        self, 
        position_traj,  # MINCO JointTrajectory for 3D position
        waypoint_times: npt.NDArray[np.float64],
        waypoint_rotations: npt.NDArray[np.float64],  # (N, 3) rotvec
        time_offset: float = 0.0,  # Offset to convert to wall clock time
    ):
        """
        Args:
            position_traj: MINCO trajectory for position (3 DOF)
            waypoint_times: (N,) relative timestamps for each waypoint
            waypoint_rotations: (N, 3) rotation vectors for each waypoint
            time_offset: Add this to convert relative times to wall clock times
        """
        self._pos_traj = position_traj
        self._waypoint_times_rel = waypoint_times
        self._waypoint_rotations = waypoint_rotations
        self._time_offset = time_offset
        
        # Create Slerp interpolator for rotation (using relative times)
        rotations = st.Rotation.from_rotvec(waypoint_rotations)
        self._rot_interp = st.Slerp(waypoint_times, rotations)
        
        # Cache duration
        self._duration = position_traj.get_total_duration()
    
    @property
    def total_duration(self) -> float:
        """Get total trajectory duration."""
        return self._duration
    
    @property
    def start_time(self) -> float:
        """Get start time (wall clock)."""
        return self._time_offset
    
    @property
    def end_time(self) -> float:
        """Get end time (wall clock)."""
        return self._time_offset + self._duration
    
    def _to_relative_time(self, t_wall: float) -> float:
        """Convert wall clock time to relative trajectory time."""
        return t_wall - self._time_offset
    
    def get_pos(self, t_wall: float) -> npt.NDArray[np.float64]:
        """Get position at wall clock time t (x, y, z)."""
        t_rel = self._to_relative_time(t_wall)
        t_rel = np.clip(t_rel, 0, self._duration)
        pos_7d = self._pos_traj.get_pos(t_rel)
        return pos_7d[:3]
    
    def get_vel(self, t_wall: float) -> npt.NDArray[np.float64]:
        """Get velocity at wall clock time t (vx, vy, vz)."""
        t_rel = self._to_relative_time(t_wall)
        t_rel = np.clip(t_rel, 0, self._duration)
        vel_7d = self._pos_traj.get_vel(t_rel)
        return vel_7d[:3]
    
    def get_acc(self, t_wall: float) -> npt.NDArray[np.float64]:
        """Get acceleration at wall clock time t."""
        t_rel = self._to_relative_time(t_wall)
        t_rel = np.clip(t_rel, 0, self._duration)
        acc_7d = self._pos_traj.get_acc(t_rel)
        return acc_7d[:3]
    
    def get_rot(self, t_wall: float) -> npt.NDArray[np.float64]:
        """Get rotation at wall clock time t (rx, ry, rz as rotvec)."""
        t_rel = self._to_relative_time(t_wall)
        t_rel = np.clip(t_rel, 0, self._duration)
        rot = self._rot_interp([t_rel])
        return rot.as_rotvec()[0]
    
    def __call__(self, t_wall: float) -> npt.NDArray[np.float64]:
        """Get full TCP pose at wall clock time t (x, y, z, rx, ry, rz)."""
        pos = self.get_pos(t_wall)
        rot = self.get_rot(t_wall)
        return np.concatenate([pos, rot])
    
    def get_tcp_pose(self, t_wall: float) -> npt.NDArray[np.float64]:
        """Alias for __call__."""
        return self(t_wall)
    
    def sample(self, dt: float = 0.005) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Sample trajectory at regular intervals.
        
        Args:
            dt: Time step for sampling (default 5ms for 200Hz)
        
        Returns:
            (timestamps, poses) where:
            - timestamps: (N,) wall clock times
            - poses: (N, 6) TCP poses [x, y, z, rx, ry, rz]
        """
        num_samples = int(np.ceil(self._duration / dt)) + 1
        timestamps = np.linspace(self.start_time, self.end_time, num_samples)
        poses = np.zeros((num_samples, 6))
        for i, t in enumerate(timestamps):
            poses[i] = self.get_tcp_pose(t)
        return timestamps, poses


class CartesianMincoOptimizer:
    """
    MINCO trajectory optimizer for Cartesian (end-effector) space.
    
    This optimizes only the position (x, y, z) using MINCO.
    Rotation is handled separately using Slerp interpolation.
    """
    
    def __init__(
        self,
        cartesian_limits: Optional[CartesianLimits] = None,
        wei_time: float = 1e2,
        wei_feas: float = 1e8,
        fix_waypoints: bool = True,
    ):
        """
        Initialize the optimizer.
        
        Args:
            cartesian_limits: Position/velocity/acceleration limits
            wei_time: Weight for time minimization (higher = more aggressive)
            wei_feas: Weight for feasibility constraints
            fix_waypoints: If True, trajectory must pass through all waypoints
        """
        if not MINCO_AVAILABLE:
            raise RuntimeError("MINCO library not available. Please build it first.")
        
        # Default Cartesian limits for the robot
        if cartesian_limits is None:
            cartesian_limits = DEFAULT_ROBOT_CARTESIAN_LIMITS
        
        self.cartesian_limits = cartesian_limits
        self.wei_time = wei_time
        self.wei_feas = wei_feas
        self.fix_waypoints = fix_waypoints
        
        self._optimizer = jto.JointTrajOptimizer()
        self._update_params()
    
    def _update_params(self):
        """Update optimizer parameters."""
        # MINCO expects 7-DOF, we use first 3 for position
        min_pos_7d = np.zeros(7)
        max_pos_7d = np.zeros(7)
        max_vel_7d = np.zeros(7)
        max_acc_7d = np.zeros(7)
        
        min_pos_7d[:3] = self.cartesian_limits.pos_min
        max_pos_7d[:3] = self.cartesian_limits.pos_max
        max_vel_7d[:3] = self.cartesian_limits.vel_max
        max_acc_7d[:3] = self.cartesian_limits.acc_max
        
        # Unused dimensions - set large limits
        for i in range(3, 7):
            min_pos_7d[i] = -100.0
            max_pos_7d[i] = 100.0
            max_vel_7d[i] = 100.0
            max_acc_7d[i] = 1000.0
        
        self._optimizer.set_param(
            min_pos=min_pos_7d,
            max_pos=max_pos_7d,
            max_vel=max_vel_7d,
            max_acc=max_acc_7d,
            wei_time=self.wei_time,
            wei_feas=self.wei_feas,
            cps_per_piece=5,
            fix_waypoints=self.fix_waypoints,
        )
    
    def _check_trajectory_feasibility(
        self,
        tcp_waypoints: npt.NDArray[np.float64],
        init_times: npt.NDArray[np.float64],
    ) -> bool:
        """
        Check if the trajectory is suitable for MINCO optimization.
        
        Returns False if trajectory has very small movements that would cause
        numerical issues in optimization.
        """
        positions = tcp_waypoints[:, :3]
        
        # Check total displacement
        total_disp = np.linalg.norm(positions[-1] - positions[0])
        if total_disp < 0.003:  # Less than 3mm total movement
            return False
        
        # Check segment displacements
        segment_disps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        
        # If any segment has very small displacement with short time, skip MINCO
        for disp, dt in zip(segment_disps, init_times):
            if disp < 0.001 and dt < 0.1:  # < 1mm in < 100ms
                return False
        
        return True
    
    def optimize(
        self,
        tcp_waypoints: npt.NDArray[np.float64],
        init_times: npt.NDArray[np.float64],
        head_vel: Optional[npt.NDArray[np.float64]] = None,
        head_acc: Optional[npt.NDArray[np.float64]] = None,
        tail_vel: Optional[npt.NDArray[np.float64]] = None,
        tail_acc: Optional[npt.NDArray[np.float64]] = None,
        start_time: float = 0.0,
        verbose: bool = False,
    ) -> Tuple[bool, Optional[CartesianMincoTrajectory], Optional[npt.NDArray[np.float64]]]:
        """
        Optimize Cartesian trajectory.
        
        Args:
            tcp_waypoints: (N, 6) TCP waypoints [x, y, z, rx, ry, rz]
            init_times: (N-1,) initial time allocation for each segment
            head_vel: (3,) initial position velocity
            head_acc: (3,) initial position acceleration
            tail_vel: (3,) final position velocity
            tail_acc: (3,) final position acceleration
            start_time: Wall clock time when trajectory starts
            verbose: Print debug info on failure (default: False)
        
        Returns:
            (success, trajectory, optimized_times)
        """
        num_waypoints = tcp_waypoints.shape[0]
        if num_waypoints < 2:
            return False, None, None
        
        # Check trajectory feasibility before optimization
        if not self._check_trajectory_feasibility(tcp_waypoints, init_times):
            return False, None, None
        
        # Extract position and rotation
        positions = tcp_waypoints[:, :3]
        rotations = tcp_waypoints[:, 3:]
        
        # Default boundary conditions
        if head_vel is None:
            head_vel = np.zeros(3)
        if head_acc is None:
            head_acc = np.zeros(3)
        if tail_vel is None:
            tail_vel = np.zeros(3)
        if tail_acc is None:
            tail_acc = np.zeros(3)
        
        # Ensure minimum time allocation to avoid numerical issues
        init_times = np.maximum(init_times, 0.05)  # At least 50ms per segment
        
        # Pad to 7-DOF
        waypoints_7d = np.zeros((num_waypoints, 7))
        waypoints_7d[:, :3] = positions
        waypoints_7d = waypoints_7d.T  # (7, N)
        
        # Build boundary conditions: (7, 4) for [pos, vel, acc, jerk]
        head_state = np.zeros((7, 4))
        tail_state = np.zeros((7, 4))
        
        head_state[:, 0] = waypoints_7d[:, 0]
        tail_state[:, 0] = waypoints_7d[:, -1]
        head_state[:3, 1] = head_vel
        tail_state[:3, 1] = tail_vel
        head_state[:3, 2] = head_acc
        tail_state[:3, 2] = tail_acc
        
        try:
            success, opt_times, pos_traj = self._optimizer.optimize(
                waypoints=waypoints_7d,
                init_times=init_times,
                head_state=head_state,
                tail_state=tail_state,
            )
            
            if not success:
                return False, None, None
            
            # Compute waypoint times (relative)
            waypoint_times = np.zeros(num_waypoints)
            waypoint_times[1:] = np.cumsum(opt_times)
            
            # Create trajectory
            traj = CartesianMincoTrajectory(
                position_traj=pos_traj,
                waypoint_times=waypoint_times,
                waypoint_rotations=rotations,
                time_offset=start_time,
            )
            
            return True, traj, opt_times
            
        except Exception as e:
            if verbose:
                print(f"[CartesianMincoOptimizer] Optimization failed: {e}")
            return False, None, None


def optimize_diffusion_trajectory(
    waypoints: npt.NDArray[np.float64],
    timestamps: npt.NDArray[np.float64],
    cartesian_limits: Optional[CartesianLimits] = None,
    current_vel: Optional[npt.NDArray[np.float64]] = None,
    verbose: bool = False,
) -> Tuple[bool, Optional[npt.NDArray[np.float64]], Optional[npt.NDArray[np.float64]]]:
    """
    High-level function to optimize diffusion policy output trajectory.
    
    This is the main entry point for MINCO post-processing of diffusion policy outputs.
    
    Args:
        waypoints: (N, 7) waypoints from diffusion policy [x, y, z, rx, ry, rz, gripper]
        timestamps: (N,) timestamps for each waypoint (wall clock time)
        cartesian_limits: Optional position/velocity/acceleration limits
        current_vel: (3,) current end-effector velocity for smooth start
        verbose: Print optimization info
    
    Returns:
        (success, sampled_poses, sampled_timestamps) where:
        - success: Whether optimization succeeded
        - sampled_poses: (M, 7) poses sampled at 200Hz [x, y, z, rx, ry, rz, gripper]
        - sampled_timestamps: (M,) timestamps for sampled poses
    """
    if not MINCO_AVAILABLE:
        if verbose:
            print("[MINCO] Not available, returning original trajectory")
        return False, None, None
    
    num_waypoints = waypoints.shape[0]
    if num_waypoints < 2:
        if verbose:
            print("[MINCO] Need at least 2 waypoints")
        return False, None, None
    
    # Extract TCP poses and gripper
    tcp_waypoints = waypoints[:, :6]
    gripper_waypoints = waypoints[:, 6]
    
    # Calculate position change
    total_pos_change = np.linalg.norm(tcp_waypoints[-1, :3] - tcp_waypoints[0, :3])
    if total_pos_change < 0.005:  # Less than 5mm movement
        if verbose:
            print(f"[MINCO] Position change too small ({total_pos_change*1000:.1f}mm), skipping")
        return False, None, None
    
    # Check per-segment displacement
    segment_disps = np.linalg.norm(np.diff(tcp_waypoints[:, :3], axis=0), axis=1)
    if np.any(segment_disps < 0.002):  # Any segment < 2mm
        if verbose:
            print(f"[MINCO] Some segments too small (min={np.min(segment_disps)*1000:.1f}mm), skipping")
        return False, None, None
    
    # Calculate initial time allocation
    init_times = np.diff(timestamps)
    init_times = np.maximum(init_times, 0.05)  # Minimum 50ms per segment
    
    if cartesian_limits is None:
        cartesian_limits = DEFAULT_ROBOT_CARTESIAN_LIMITS
    
    try:
        optimizer = CartesianMincoOptimizer(
            cartesian_limits=cartesian_limits,
            wei_time=1e2,    # Reduced for stability
            wei_feas=1e8,    # Higher for constraint satisfaction
            fix_waypoints=True,
        )
        
        head_vel = current_vel if current_vel is not None else np.zeros(3)
        
        success, minco_traj, opt_times = optimizer.optimize(
            tcp_waypoints=tcp_waypoints,
            init_times=init_times,
            head_vel=head_vel,
            start_time=timestamps[0],
            verbose=verbose,
        )
        
        if not success or minco_traj is None:
            if verbose:
                print("[MINCO] Optimization failed")
            return False, None, None
        
        # Sample at 200Hz
        sample_dt = 0.005  # 5ms = 200Hz
        sampled_timestamps, sampled_tcp_poses = minco_traj.sample(dt=sample_dt)
        
        # Interpolate gripper positions
        sampled_gripper = np.interp(sampled_timestamps, timestamps, gripper_waypoints)
        
        # Combine TCP poses and gripper
        sampled_poses = np.zeros((len(sampled_timestamps), 7))
        sampled_poses[:, :6] = sampled_tcp_poses
        sampled_poses[:, 6] = sampled_gripper
        
        if verbose:
            original_duration = timestamps[-1] - timestamps[0]
            optimized_duration = minco_traj.total_duration
            time_change = (optimized_duration - original_duration) / original_duration * 100
            print(f"[MINCO] Optimization successful:")
            print(f"  Waypoints: {num_waypoints} -> Samples: {len(sampled_timestamps)}")
            print(f"  Original duration:  {original_duration:.3f}s")
            print(f"  Optimized duration: {optimized_duration:.3f}s")
            print(f"  Time change: {time_change:+.1f}%")
        
        return True, sampled_poses, sampled_timestamps
        
    except Exception as e:
        if verbose:
            print(f"[MINCO] Exception during optimization: {e}")
        return False, None, None


# ============ Test Functions ============

def test_minco_optimizer():
    """Test the MINCO trajectory optimizer."""
    print("Testing CartesianMincoOptimizer...")
    
    if not MINCO_AVAILABLE:
        print("MINCO not available, skipping test")
        return False
    
    # Create test waypoints (6 DOF TCP poses)
    num_waypoints = 5
    waypoints = np.zeros((num_waypoints, 6))
    waypoints[0] = [0.4, 0.0, 0.3, 0.0, 3.14, 0.0]
    waypoints[1] = [0.45, 0.05, 0.32, 0.0, 3.14, 0.1]
    waypoints[2] = [0.5, 0.0, 0.35, 0.0, 3.14, 0.0]
    waypoints[3] = [0.45, -0.05, 0.32, 0.0, 3.14, -0.1]
    waypoints[4] = [0.4, 0.0, 0.3, 0.0, 3.14, 0.0]
    
    init_times = np.ones(num_waypoints - 1) * 0.3
    
    optimizer = CartesianMincoOptimizer()
    success, traj, opt_times = optimizer.optimize(
        tcp_waypoints=waypoints,
        init_times=init_times,
        start_time=0.0,
    )
    
    print(f"Optimization success: {success}")
    
    if success:
        print(f"Optimized times: {opt_times}")
        print(f"Total duration: {traj.total_duration:.3f}s")
        
        # Sample trajectory
        timestamps, poses = traj.sample(dt=0.01)
        print(f"Sampled poses shape: {poses.shape}")
        
        # Verify waypoints
        print("\nWaypoint verification:")
        wp_times = np.zeros(num_waypoints)
        wp_times[1:] = np.cumsum(opt_times)
        for i, t in enumerate(wp_times):
            pos = traj.get_pos(t)
            error = np.linalg.norm(pos - waypoints[i, :3])
            print(f"  Waypoint {i} at t={t:.3f}s: position error = {error*1000:.2f}mm")
    
    print("\nTest completed!")
    return success


if __name__ == "__main__":
    test_minco_optimizer()
