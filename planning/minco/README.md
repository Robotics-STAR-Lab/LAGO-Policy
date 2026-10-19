# Joint Space Trajectory Optimization (MINCO) for Robotic Arms

A C++/Python library for joint-space trajectory optimization of robotic arms, implementing the Minimum Snap (MINCO) trajectory optimization algorithm.

## Features

- **Minimum-snap optimization**: smooth trajectories that minimize snap using degree-7 polynomials
- **Time optimization**: automatic optimization of the per-segment time allocation
- **Constraint handling**: supports joint position, velocity, and acceleration limits
- **Efficient solver**: fast convergence with the L-BFGS optimizer
- **Python bindings**: Python interface via pybind11
- **No ROS dependency**: pure C++ implementation, usable standalone

## Dependencies

- C++17
- Eigen3 (`sudo apt install libeigen3-dev`)
- Python 3.8+
- pybind11 (`pip install pybind11`)
- numpy (Python)

## Building

First activate the project's conda environment (e.g. `conda activate lago_policy`) so that the correct Python is used and pybind11 can be found.

### Option 1: use the build.sh script (recommended)

```bash
cd planning/minco
bash build.sh
```

### Option 2: manual build

```bash
cd planning/minco
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(which python)"
make -j$(nproc)
```

After building, the `joint_traj_opt.cpython-*.so` file is generated in the `build` directory.

## Usage

### Python API

```python
import numpy as np
import sys
sys.path.insert(0, '/path/to/planning/minco/build')
import joint_traj_opt as jto

# Define waypoints: start + 6 intermediate points + end = 8 points
dof = 7
num_waypoints = 8

waypoints = np.zeros((dof, num_waypoints))
waypoints[:, 0] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # start
waypoints[:, -1] = [0.5, 0, 0.5, -1.5, 0, 2.0, 0.5]  # end
# ... set the intermediate waypoints ...

# Initial time allocation (1 s per segment)
init_times = np.ones(num_waypoints - 1)

# Option 1: quick optimization with the default joint parameters
# (the trajectory is forced to pass through all waypoints)
success, opt_times, traj = jto.optimize_franka_trajectory_default(
    waypoints, init_times, fix_waypoints=True)

# Option 2: custom parameters
optimizer = jto.JointTrajOptimizer()
optimizer.set_param(
    min_pos=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    max_pos=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
    max_vel=np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]),
    max_acc=np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]),
    wei_time=1.0,       # time weight
    wei_feas=1e3,       # feasibility weight
    fix_waypoints=True  # True = force the trajectory through all waypoints; False = allow optimizing intermediate positions
)

head_state = np.zeros((dof, 4))  # [pos, vel, acc, jerk]
tail_state = np.zeros((dof, 4))
head_state[:, 0] = waypoints[:, 0]
tail_state[:, 0] = waypoints[:, -1]

success, opt_times, traj = optimizer.optimize(waypoints, init_times, head_state, tail_state)

# Sample the trajectory
if success:
    positions = traj.sample(dt=0.01, derivative_order=0)  # position
    velocities = traj.sample(dt=0.01, derivative_order=1)  # velocity
    accelerations = traj.sample(dt=0.01, derivative_order=2)  # acceleration

    # Or query a specific time
    pos_t = traj.get_pos(0.5)  # position at t = 0.5 s
    vel_t = traj.get_vel(0.5)  # velocity at t = 0.5 s
```

### Using the MinSnapOpt class (low-level API)

```python
import joint_traj_opt as jto

# Create the optimizer
minsnap = jto.MinSnapOpt()
minsnap.reset(piece_num=7)  # 7 trajectory segments

# Set boundary conditions and intermediate points
head_state = np.zeros((7, 4))  # [pos, vel, acc, jerk]
tail_state = np.zeros((7, 4))
inner_pts = np.zeros((7, 6))  # 6 intermediate points

# Generate the trajectory
times = np.ones(7)  # time allocation
minsnap.generate(inner_pts, times, head_state, tail_state)

# Get the cost (note: the method is named get_traj_jerk_cost but it is actually the snap cost)
cost = minsnap.get_traj_jerk_cost()

# Get the trajectory
traj = minsnap.get_traj()
```

## Examples

Run the quick test (validates basic module functionality):

```bash
cd planning/minco/examples
python quick_test.py
```

## API reference

### JointTrajOptimizer

| Method | Description |
|------|------|
| `set_param(min_pos, max_pos, max_vel, max_acc, wei_time, wei_feas, cps_per_piece, fix_waypoints)` | Set the optimization parameters |
| `set_fix_waypoints(fix)` | Set whether to fix the intermediate waypoints |
| `set_default_franka_param()` | Use the default joint parameters (Franka Emika Panda limits) |
| `optimize(waypoints, init_times, head_state, tail_state)` | Run optimization, returns (success, opt_times, trajectory) |
| `get_iter_num()` | Get the number of iterations |

**Notes on `fix_waypoints`:**
- `fix_waypoints=False` (default): optimize intermediate positions + time; smoother trajectory, but not guaranteed to pass through all waypoints
- `fix_waypoints=True`: fix the intermediate positions and only optimize time; **the trajectory is forced through all waypoints**

### JointTrajectory

| Method | Description |
|------|------|
| `get_piece_num()` | Get the number of trajectory segments |
| `get_dim()` | Get the number of degrees of freedom |
| `get_total_duration()` | Get the total duration |
| `get_durations()` | Get the duration of each segment |
| `get_pos(t)` | Get the position at time t |
| `get_vel(t)` | Get the velocity at time t |
| `get_acc(t)` | Get the acceleration at time t |
| `get_jerk(t)` | Get the jerk at time t |
| `sample(dt, derivative_order)` | Sample the trajectory |

### MinSnapOpt

| Method | Description |
|------|------|
| `reset(piece_num)` | Reset the optimizer |
| `generate(inner_pts, times, head_state, tail_state)` | Generate a trajectory |
| `get_traj_jerk_cost()` | Get the trajectory cost (actually the snap cost) |
| `get_traj()` | Get the optimized trajectory |
| `get_init_constrain_points(K)` | Get the constraint points |

### Convenience functions

| Function | Description |
|------|------|
| `optimize_franka_trajectory(...)` | Optimize with custom parameters |
| `optimize_franka_trajectory_default(waypoints, init_times)` | Optimize with the default joint parameters |

## File structure

```
planning/minco/
├── CMakeLists.txt           # build configuration
├── build.sh                 # build script
├── README.md                # this document
├── src/
│   └── joint_traj_bindings.cpp  # pybind11 bindings
├── traj_opt/
│   └── include/
│       └── optimizer/
│           ├── lbfgs.hpp             # L-BFGS optimizer
│           └── poly_traj_optimizer.hpp  # trajectory optimizer
├── traj_utils/
│   └── include/
│       └── traj_utils/
│           ├── poly_traj_utils.hpp   # trajectory utilities (MinSnapOpt)
│           └── root_finder.hpp       # polynomial root finding
└── examples/
    └── quick_test.py           # quick test
```

## Algorithm notes

This implementation is based on the MINCO (Minimum Control Effort) trajectory optimization method:

- **Polynomial degree**: degree-7 polynomials (degree = 7)
- **Objective**: minimize the integral of snap (the fourth derivative)
- **Boundary conditions**: position, velocity, acceleration, and jerk at the start and end
- **Constraints**: joint position limits, joint velocity limits, joint acceleration limits
- **Solver**: L-BFGS quasi-Newton optimizer

Note: although some method names contain "Jerk" (e.g. `getTrajJerkCost`), the quantity actually optimized is the snap cost; the names are kept consistent with the original code.

## License

MIT License
