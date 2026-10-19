"""
Goal-directed collision-free planning glue for LAGO Policy.

This module is a hardware-free extraction of the planning logic used at
deployment time. It connects the goal predicted by ``GoalAwareDiffusionPolicy``
to the EGO-Planner-style B-spline trajectory optimizer (built as the
``ego_planner_cpp`` pybind module under ``planning/ego_planner/``).

It contains:
  * ``EgoPlannerManager``      - thin wrapper around the C++ B-spline optimizer
                                 (obstacle set, physical limits, A* guiding path
                                 + signed-distance collision penalty).
  * ``GoalDirectedModeSelector`` - distance-threshold + hysteresis switch that
                                 decides whether to navigate to the predicted
                                 goal (collision-free planner) or to execute the
                                 raw diffusion action chunk.
  * ``TrajectoryStitcher``     - predicts the boundary state of the previous
                                 chunk for smooth, latency-free chunk stitching.
  * helper functions ``goal_to_pose`` and ``interpolate_pose_by_time``.

All real-robot / camera / ZMQ coupling has been intentionally removed; the
classes operate purely on numpy arrays so they can be reused with any robot
back-end or in simulation.
"""

import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from diffusion_policy.common.pose_util import rot6d_to_mat


# --------------------------------------------------------------------------- #
# Locate and import the compiled ego-planner pybind module (ego_planner_cpp).
# Build it first with: cd planning/ego_planner && bash build.sh
# (the build emits ego_planner_cpp*.so into planning/ego_planner/build).
# --------------------------------------------------------------------------- #
_EGO_PLANNER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ego_planner", "build")
if os.path.isdir(_EGO_PLANNER_DIR) and _EGO_PLANNER_DIR not in sys.path:
    sys.path.insert(0, _EGO_PLANNER_DIR)

try:
    import ego_planner_cpp as ego
    EGO_PLANNER_AVAILABLE = True
except ImportError as e:  # pragma: no cover - depends on a compiled extension
    print(f"Warning: ego_planner_cpp not available: {e}")
    print("  Build it with: cd planning/ego_planner && bash build.sh")
    EGO_PLANNER_AVAILABLE = False


# Default ego-planner physical limits (tuned for a table-top manipulator).
EGO_MAX_VEL = 0.2      # m/s
EGO_MAX_ACC = 1.0      # m/s^2
EGO_SAFE_DIST = 0.1    # m, safety clearance from obstacles
# Waypoint sampling interval for the planned trajectory (50 Hz).
SAMPLING_INTERVAL = 0.02


class EgoPlannerManager:
    """Wrapper around the EGO-Planner B-spline trajectory optimizer.

    Provides obstacle-aware, collision-free trajectory planning from the
    current end-effector pose to the predicted goal. Internally the C++ module
    runs an A* search for a collision-free guiding path and then refines a
    uniform B-spline by minimizing a weighted sum of smoothness (jerk),
    a signed-distance collision penalty, and dynamic-feasibility terms.
    """

    def __init__(
        self,
        obstacles: Optional[list] = None,
        max_vel: float = EGO_MAX_VEL,
        max_acc: float = EGO_MAX_ACC,
        safe_dist: float = EGO_SAFE_DIST,
        ground_level: float = 0.05,
        workspace_min: Optional[np.ndarray] = None,
        workspace_max: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """
        Args:
            obstacles: list of ``(bmin, bmax)`` AABB obstacle corners.
            max_vel: maximum velocity (m/s).
            max_acc: maximum acceleration (m/s^2).
            safe_dist: safety clearance from obstacles (m).
            ground_level: minimum Z level (table surface) used by the A* search.
            workspace_min/max: workspace bounds (pre-allocated GridMap).
            verbose: enable C++ debug output.
        """
        if not EGO_PLANNER_AVAILABLE:
            raise RuntimeError(
                "ego_planner_cpp not available. Build it first "
                "(planning/ego_planner/build.sh).")

        self.optimizer = ego.BsplineOptimizer()
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.safe_dist = safe_dist
        self.verbose = verbose

        self.optimizer.set_physical_limits(
            max_vel=max_vel, max_acc=max_acc, safe_dist=safe_dist)
        self.optimizer.set_weights(
            smooth=10.0, collision=50000.0, feasibility=1.0, fitness=1.0)
        self.optimizer.set_ground_level(ground_level)
        self.optimizer.set_prefer_vertical(True, discount=0.1)
        self.optimizer.set_verbose(verbose)

        if workspace_min is None:
            workspace_min = np.array([0.1, -0.5, 0.05])
        if workspace_max is None:
            workspace_max = np.array([0.7, 0.5, 0.5])
        self.optimizer.set_workspace(workspace_min, workspace_max)

        self.obstacles = []
        if obstacles is not None:
            for bmin, bmax in obstacles:
                self.add_obstacle(bmin, bmax)

    def add_obstacle(self, bmin: np.ndarray, bmax: np.ndarray):
        """Add an axis-aligned bounding box (AABB) obstacle."""
        bmin = np.asarray(bmin, dtype=np.float64)
        bmax = np.asarray(bmax, dtype=np.float64)
        self.optimizer.add_obstacle(bmin, bmax)
        self.obstacles.append((bmin.copy(), bmax.copy()))
        if self.verbose:
            print(f"[EgoPlanner] Added obstacle: [{bmin}] to [{bmax}]")

    def clear_obstacles(self):
        """Clear all obstacles."""
        self.optimizer.clear_obstacles()
        self.obstacles.clear()

    def plan_trajectory(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        num_ctrl_pts: int = 20,
        ts: float = 0.05,
        max_iterations: int = 150,
    ) -> tuple:
        """Plan a collision-free trajectory from ``start_pos`` to ``goal_pos``.

        Returns ``(success, trajectory, sampled_positions)`` where
        ``sampled_positions`` is a ``(3, N)`` array sampled at
        ``SAMPLING_INTERVAL``.
        """
        start_pos = np.asarray(start_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)

        try:
            trajectory = self.optimizer.optimize(
                start_pos, goal_pos,
                num_ctrl_pts=num_ctrl_pts, ts=ts, max_iterations=max_iterations)
            sampled_positions = trajectory.sample_trajectory(SAMPLING_INTERVAL)
            if self.verbose:
                print(f"[EgoPlanner] duration={trajectory.get_duration():.2f}s, "
                      f"length={trajectory.get_length():.3f}m, "
                      f"samples={sampled_positions.shape[1]}")
            return True, trajectory, sampled_positions
        except Exception as e:  # pragma: no cover
            if self.verbose:
                print(f"[EgoPlanner] Planning failed: {e}")
            return False, None, None

    def generate_commands_with_rotation(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        start_rot_mat: np.ndarray,
        goal_rot_mat: np.ndarray,
        start_gripper: float,
        goal_gripper: float,
    ) -> tuple:
        """Plan position with the ego-planner and add SLERP rotation + gripper.

        Returns ``(commands, duration)`` where each command is
        ``[x, y, z, rx, ry, rz, gripper]`` (rotation as axis-angle), or
        ``(None, None)`` if planning fails.
        """
        success, trajectory, sampled_positions = self.plan_trajectory(start_pos, goal_pos)
        if not success or sampled_positions is None:
            return None, None

        duration = trajectory.get_duration()
        n_samples = sampled_positions.shape[1]

        key_rots = R.from_matrix(np.stack([start_rot_mat, goal_rot_mat]))
        slerp = Slerp([0, 1], key_rots)

        start_gripper = float(np.asarray(start_gripper).flatten()[0])
        goal_gripper = float(np.asarray(goal_gripper).flatten()[0])

        commands = []
        last_pos = None
        min_pos_change = 0.001  # drop near-duplicate waypoints (1mm)
        for i in range(n_samples):
            alpha = i / max(n_samples - 1, 1)
            pos = sampled_positions[:, i]
            if last_pos is not None and i < n_samples - 1:
                if np.linalg.norm(pos - last_pos) < min_pos_change:
                    continue
            rot = slerp(alpha).as_rotvec()
            gripper = start_gripper * (1 - alpha) + goal_gripper * alpha
            commands.append(np.concatenate([pos, rot, [gripper]]))
            last_pos = pos
        return commands, duration


def goal_to_pose(goal: np.ndarray) -> dict:
    """Convert a predicted goal ``[pos(3), rot6d(6), gripper(1)]`` to a pose dict.

    Returns ``{'position', 'rotation_matrix', 'gripper'}``.
    """
    pos = goal[:3]
    rot6d = goal[3:9]
    gripper = goal[9]
    rot_mat = rot6d_to_mat(rot6d.reshape(1, 6))[0]
    return {"position": pos, "rotation_matrix": rot_mat, "gripper": gripper}


def interpolate_pose_by_time(
    current_pos, current_rot_mat, target_pos, target_rot_mat,
    current_gripper, target_gripper,
    duration: float,
    sample_interval: float = SAMPLING_INTERVAL,
):
    """Linear position + SLERP rotation interpolation (fallback planner).

    Returns ``(commands, duration)`` with each command formatted as
    ``[x, y, z, rx, ry, rz, gripper]``.
    """
    n_samples = max(2, int(np.ceil(duration / sample_interval)) + 1)
    key_rots = R.from_matrix(np.stack([current_rot_mat, target_rot_mat]))
    slerp = Slerp([0, 1], key_rots)
    current_gripper = float(np.asarray(current_gripper).flatten()[0])
    target_gripper = float(np.asarray(target_gripper).flatten()[0])

    commands = []
    for i in range(n_samples):
        alpha = i / max(n_samples - 1, 1)
        interp_pos = current_pos * (1 - alpha) + target_pos * alpha
        interp_rot = slerp(alpha).as_rotvec()
        interp_gripper = current_gripper * (1 - alpha) + target_gripper * alpha
        commands.append(np.concatenate([interp_pos, interp_rot, [interp_gripper]]))
    return commands, duration


class GoalDirectedModeSelector:
    """Decide between collision-free navigation and raw action-chunk execution.

    Mirrors the deployment trigger logic of LAGO Policy. There is **no**
    explicit line-of-sight ray-cast: collision avoidance is delegated to the
    EGO-Planner (A* guiding path + signed-distance penalty). The trigger is a
    distance-to-goal threshold with hysteresis to avoid mode chattering:

      * enter NAV  (plan to goal)  if ``dist > nav_threshold_high``
      * exit  NAV  (execute chunk) if ``dist < nav_threshold_low`` and
        current EE speed ``< action_velocity_threshold``

    If two consecutive predicted goals are closer than ``min_goal_dist`` (and
    the current distance is already small) the raw action chunk is used.
    """

    def __init__(
        self,
        nav_threshold_high: float = 0.05,
        nav_threshold_low: float = 0.05,
        action_velocity_threshold: float = 0.5,
        min_goal_dist: float = 0.2,
    ):
        self.nav_threshold_high = nav_threshold_high
        self.nav_threshold_low = nav_threshold_low
        self.action_velocity_threshold = action_velocity_threshold
        self.min_goal_dist = min_goal_dist
        self._in_nav_mode = False
        self._last_goal_pos: Optional[np.ndarray] = None

    def reset(self):
        """Reset hysteresis state at the start of an episode."""
        self._in_nav_mode = False
        self._last_goal_pos = None

    def decide(self, current_pos: np.ndarray, goal_pos: np.ndarray,
               ee_velocity: float) -> str:
        """Return ``"NAV"`` (collision-free planning) or ``"ACTION"`` (chunk).

        Args:
            current_pos: current end-effector position (3,).
            goal_pos: predicted goal position (3,).
            ee_velocity: current end-effector speed (m/s).
        """
        dist = float(np.linalg.norm(current_pos - goal_pos))

        goal_change_dist = 0.0
        if self._last_goal_pos is not None:
            goal_change_dist = float(np.linalg.norm(goal_pos - self._last_goal_pos))
        use_action_for_close_goals = (
            self._last_goal_pos is not None
            and 0.001 < goal_change_dist < self.min_goal_dist
            and dist <= self.nav_threshold_high)

        if use_action_for_close_goals:
            mode = "ACTION"
        elif dist > self.nav_threshold_high:
            mode = "NAV"
            self._in_nav_mode = True
        elif self._in_nav_mode:
            if dist < self.nav_threshold_low and ee_velocity < self.action_velocity_threshold:
                mode = "ACTION"
                self._in_nav_mode = False
            else:
                mode = "NAV"
        else:
            mode = "ACTION"

        self._last_goal_pos = np.asarray(goal_pos).copy()
        return mode


class TrajectoryStitcher:
    """Predict the boundary state of the previous chunk for smooth stitching.

    Instead of using latency-prone feedback, the expected position / velocity /
    acceleration at the current time is computed from the previously commanded
    chunk. This provides continuous boundary conditions for trajectory
    optimization across inference chunks.
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.last_commands = None
        self.last_timestamps = None
        self.last_chunk_start_time = None
        self.last_action_base_pose = None
        self.last_velocities = None

    def save_chunk(self, commands: np.ndarray, timestamps: np.ndarray,
                   action_base_pose: np.ndarray, chunk_start_time: float):
        """Store the current chunk and pre-compute per-segment velocities."""
        self.last_commands = commands.copy()
        self.last_timestamps = timestamps.copy()
        self.last_action_base_pose = action_base_pose.copy()
        self.last_chunk_start_time = chunk_start_time

        if len(commands) >= 2:
            vels = []
            for i in range(len(commands) - 1):
                dt_seg = timestamps[i + 1] - timestamps[i]
                vel = (commands[i + 1, :3] - commands[i, :3]) / dt_seg if dt_seg > 0.001 else np.zeros(3)
                vels.append(vel)
            self.last_velocities = np.array(vels)
        else:
            self.last_velocities = None

    def get_predicted_state(self, current_time: float):
        """Return ``(pos, rot_axis_angle, vel, acc, valid)`` at ``current_time``."""
        if self.last_commands is None or self.last_timestamps is None:
            return None, None, None, None, False

        t0 = self.last_timestamps[0]
        t_end = self.last_timestamps[-1]
        if current_time < t0:
            return None, None, None, None, False

        if current_time > t_end + 0.1:  # past end (with 100ms grace)
            last_cmd = self.last_commands[-1]
            vel = (self.last_velocities[-1].copy()
                   if self.last_velocities is not None and len(self.last_velocities) > 0
                   else np.zeros(3))
            acc = np.zeros(3)
            if self.last_velocities is not None and len(self.last_velocities) >= 2:
                dt_seg = self.last_timestamps[-1] - self.last_timestamps[-2]
                if dt_seg > 0.001:
                    acc = (self.last_velocities[-1] - self.last_velocities[-2]) / dt_seg
            return last_cmd[:3], last_cmd[3:6], vel, acc, True

        for i in range(len(self.last_timestamps) - 1):
            t_i = self.last_timestamps[i]
            t_next = self.last_timestamps[i + 1]
            if t_i <= current_time <= t_next:
                alpha = (current_time - t_i) / (t_next - t_i) if (t_next - t_i) > 0.001 else 1.0
                cmd_i = self.last_commands[i]
                cmd_next = self.last_commands[i + 1]
                predicted_pos = cmd_i[:3] + alpha * (cmd_next[:3] - cmd_i[:3])

                key_rots = R.from_matrix(np.stack([
                    R.from_rotvec(cmd_i[3:6]).as_matrix(),
                    R.from_rotvec(cmd_next[3:6]).as_matrix()]))
                predicted_rot_aa = Slerp([0, 1], key_rots)(alpha).as_rotvec()

                vel = (self.last_velocities[i].copy()
                       if self.last_velocities is not None else np.zeros(3))
                acc = np.zeros(3)
                if self.last_velocities is not None and len(self.last_velocities) >= 2:
                    if i < len(self.last_velocities) - 1:
                        dt_next = self.last_timestamps[i + 1] - self.last_timestamps[i]
                        if dt_next > 0.001:
                            acc = (self.last_velocities[i + 1] - self.last_velocities[i]) / dt_next
                    else:
                        dt_seg = self.last_timestamps[i] - self.last_timestamps[i - 1]
                        if dt_seg > 0.001:
                            acc = (self.last_velocities[i] - self.last_velocities[i - 1]) / dt_seg
                return predicted_pos, predicted_rot_aa, vel, acc, True

        last_cmd = self.last_commands[-1]
        vel = (self.last_velocities[-1].copy()
               if self.last_velocities is not None and len(self.last_velocities) > 0
               else np.zeros(3))
        return last_cmd[:3], last_cmd[3:6], vel, np.zeros(3), True

    def reset(self):
        """Reset stitcher state."""
        self.last_commands = None
        self.last_timestamps = None
        self.last_chunk_start_time = None
        self.last_action_base_pose = None
        self.last_velocities = None
