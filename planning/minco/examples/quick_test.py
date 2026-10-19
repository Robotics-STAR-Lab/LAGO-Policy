#!/usr/bin/env python3
"""
Quick test script -- verify the basic functionality of the joint_traj_opt module.
"""

import numpy as np
import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

import joint_traj_opt as jto


def test_basic():
    """Basic functionality test."""
    print("Testing basic functionality...")
    
    # Quick optimization using the default joint parameters
    dof = 7
    
    # start + 6 intermediate points + end = 8 waypoints
    waypoints = np.zeros((dof, 8))
    waypoints[:, 0] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]  # start
    waypoints[:, 7] = [0.5, 0.0, 0.5, -1.5, 0.0, 2.0, 0.5]  # end
    
    # linearly interpolate the intermediate points
    for i in range(1, 7):
        alpha = i / 7.0
        waypoints[:, i] = (1 - alpha) * waypoints[:, 0] + alpha * waypoints[:, 7]
    
    # initial time allocation
    init_times = np.ones(7) * 0.5  # 7 segments, 0.5 s each
    
    # optimize with default parameters
    success, opt_times, traj = jto.optimize_franka_trajectory_default(waypoints, init_times)
    
    assert success, "Optimization failed!"
    assert traj.get_piece_num() == 7, "Wrong piece number"
    assert traj.get_dim() == 7, "Wrong DOF"
    
    # verify the trajectory start and end positions
    start_pos = traj.get_pos(0.0)
    end_pos = traj.get_pos(traj.get_total_duration() - 0.001)
    
    np.testing.assert_array_almost_equal(start_pos, waypoints[:, 0], decimal=5,
                                         err_msg="Start position mismatch")
    np.testing.assert_array_almost_equal(end_pos, waypoints[:, 7], decimal=3,
                                         err_msg="End position mismatch")
    
    # verify that the start velocity is zero
    start_vel = traj.get_vel(0.0)
    np.testing.assert_array_almost_equal(start_vel, np.zeros(dof), decimal=5,
                                         err_msg="Start velocity should be zero")
    
    print(f"  Total duration: {traj.get_total_duration():.3f}s")
    print("  ✓ Basic test passed!")


def test_minsnap_opt():
    """Test the MinSnapOpt class."""
    print("Testing MinSnapOpt class...")
    
    dof = 7
    piece_num = 4
    
    minsnap = jto.MinSnapOpt()
    minsnap.reset(piece_num)
    
    head_state = np.zeros((dof, 4))
    tail_state = np.zeros((dof, 4))
    head_state[:, 0] = [0, 0, 0, -1.5, 0, 1, 0]
    tail_state[:, 0] = [0.5, 0.5, 0.5, -1.0, 0.5, 1.5, 0.5]
    
    inner_pts = np.zeros((dof, piece_num - 1))
    for i in range(piece_num - 1):
        alpha = (i + 1) / piece_num
        inner_pts[:, i] = (1 - alpha) * head_state[:, 0] + alpha * tail_state[:, 0]
    
    times = np.ones(piece_num) * 0.8
    
    minsnap.generate(inner_pts, times, head_state, tail_state)
    
    cost = minsnap.get_traj_jerk_cost()
    assert cost > 0, "Cost should be positive"
    
    traj = minsnap.get_traj()
    assert traj.get_piece_num() == piece_num, "Piece number mismatch"
    
    pts = minsnap.get_init_constrain_points(K=3)
    expected_pts = piece_num * 3 + 1
    assert pts.shape == (dof, expected_pts), f"Expected shape ({dof}, {expected_pts}), got {pts.shape}"
    
    print(f"  Cost: {cost:.2f}")
    print("  ✓ MinSnapOpt test passed!")


def test_trajectory_sampling():
    """Test trajectory sampling."""
    print("Testing trajectory sampling...")
    
    # create a simple trajectory
    waypoints = np.zeros((7, 3))
    waypoints[:, 0] = [0, 0, 0, -1.5, 0, 1, 0]
    waypoints[:, 1] = [0.25, 0.25, 0.25, -1.25, 0.25, 1.25, 0.25]
    waypoints[:, 2] = [0.5, 0.5, 0.5, -1.0, 0.5, 1.5, 0.5]
    
    init_times = np.array([0.5, 0.5])
    
    success, _, traj = jto.optimize_franka_trajectory_default(waypoints, init_times)
    assert success
    
    # test sampling
    dt = 0.01
    positions = traj.sample(dt=dt, derivative_order=0)
    velocities = traj.sample(dt=dt, derivative_order=1)
    accelerations = traj.sample(dt=dt, derivative_order=2)
    jerks = traj.sample(dt=dt, derivative_order=3)
    
    expected_samples = int(traj.get_total_duration() / dt) + 1
    assert len(positions) == expected_samples, f"Expected {expected_samples} samples, got {len(positions)}"
    
    assert positions.shape[1] == 7, "Position should have 7 columns"
    assert velocities.shape[1] == 7, "Velocity should have 7 columns"
    assert accelerations.shape[1] == 7, "Acceleration should have 7 columns"
    assert jerks.shape[1] == 7, "Jerk should have 7 columns"
    
    print(f"  Samples: {len(positions)}")
    print("  ✓ Sampling test passed!")


def test_optimizer_class():
    """Test the JointTrajOptimizer class."""
    print("Testing JointTrajOptimizer class...")
    
    optimizer = jto.JointTrajOptimizer()
    
    # custom parameters
    min_pos = np.array([-2.5, -1.5, -2.5, -2.8, -2.5, -0.01, -2.5])
    max_pos = np.array([2.5, 1.5, 2.5, -0.1, 2.5, 3.5, 2.5])
    max_vel = np.array([2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5])
    max_acc = np.array([10.0, 5.0, 8.0, 10.0, 12.0, 15.0, 15.0])
    
    optimizer.set_param(min_pos, max_pos, max_vel, max_acc, 
                        wei_time=0.5, wei_feas=500)
    
    # define waypoints
    waypoints = np.zeros((7, 4))
    waypoints[:, 0] = [0, -0.5, 0, -2.0, 0, 1.0, 0]
    waypoints[:, 1] = [0.3, 0, 0.3, -1.5, 0.3, 1.5, 0.3]
    waypoints[:, 2] = [0.6, 0.5, 0.6, -1.0, 0.6, 2.0, 0.6]
    waypoints[:, 3] = [0.9, 0, 0.9, -0.5, 0.9, 2.5, 0.9]
    
    head_state = np.zeros((7, 4))
    tail_state = np.zeros((7, 4))
    head_state[:, 0] = waypoints[:, 0]
    tail_state[:, 0] = waypoints[:, -1]
    
    init_times = np.ones(3) * 0.6
    
    success, opt_times, traj = optimizer.optimize(waypoints, init_times, 
                                                   head_state, tail_state)
    
    assert success, "Custom optimization failed"
    assert optimizer.get_iter_num() > 0, "Should have iterations"
    
    print(f"  Iterations: {optimizer.get_iter_num()}")
    print(f"  Duration: {traj.get_total_duration():.3f}s")
    print("  ✓ Optimizer class test passed!")


if __name__ == "__main__":
    print("=" * 50)
    print("Joint Trajectory Optimization Quick Tests")
    print("=" * 50)
    print()
    
    try:
        test_basic()
        test_minsnap_opt()
        test_trajectory_sampling()
        test_optimizer_class()
        
        print()
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


