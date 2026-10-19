"""Goal-directed collision-free planning for LAGO Policy.

Public API:
    EgoPlannerManager          - EGO-Planner B-spline trajectory optimizer wrapper
    GoalDirectedModeSelector   - distance-threshold + hysteresis trigger
    TrajectoryStitcher         - smooth inter-chunk boundary prediction
    goal_to_pose               - decode predicted goal -> pose dict
    interpolate_pose_by_time   - linear/SLERP fallback planner
    EGO_PLANNER_AVAILABLE      - whether the ego_planner_cpp module is importable

The MINCO spatial-temporal optimizer (CartesianMincoOptimizer) lives in
``planning.minco_trajectory_optimizer`` and requires the external
``joint_traj_opt`` library (see README).
"""

from planning.ego_planner_manager import (
    EGO_PLANNER_AVAILABLE,
    EgoPlannerManager,
    GoalDirectedModeSelector,
    TrajectoryStitcher,
    goal_to_pose,
    interpolate_pose_by_time,
)

__all__ = [
    "EGO_PLANNER_AVAILABLE",
    "EgoPlannerManager",
    "GoalDirectedModeSelector",
    "TrajectoryStitcher",
    "goal_to_pose",
    "interpolate_pose_by_time",
]
