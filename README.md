# LAGO Policy

**Latency-Aware Asynchronous Diffusion Policies with Goal-Directed Collision-Free Planning for Smooth Manipulation**

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://lago-policy.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.17982-b31b1b.svg)](https://arxiv.org/abs/2606.17982)

Guowei Shi, Xupeng Xie, Yiming Luo, Jian Guo, Jun Ma, Boyu Zhou

## Overview

LAGO Policy is a unified asynchronous action-generation framework that combines a diffusion policy with model-based trajectory optimization for smooth, temporally consistent, and collision-free manipulation. It builds on UMI's [Diffusion Policy](https://github.com/real-stanford/universal_manipulation_interface), and the planning module draws on [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner) with [MINCO](https://github.com/ZJU-FAST-Lab/GCOPTER)-style spatial-temporal optimization. This repository contains the model/algorithm code (training, architecture, and planning).

- **Latency-Aware CFG (LA-CFG):** conditions on future actions to smooth inter-chunk boundaries under asynchronous inference.
- **Goal-prediction head:** predicts a task-relevant interaction goal from demonstrations.
- **Goal-directed planning:** collision-free B-spline planning + spatial-temporal optimization for low-jerk, safe motion.

## Installation

```bash
conda env create -f conda_environment.yaml
conda activate lago_policy
```

Build the two C++ pybind11 modules (both need Eigen3 + pybind11: `sudo apt install libeigen3-dev && pip install pybind11`):

```bash
# Collision-free B-spline planner (ego-planner)
(cd planning/ego_planner && bash build.sh)

# MINCO spatial-temporal optimizer (bundled under planning/minco/)
(cd planning/minco && bash build.sh)
```

The MINCO optimizer is **included in this repo** (`planning/minco/`) and built from source just like ego-planner. After building, `planning/minco_trajectory_optimizer.py` finds it automatically at `planning/minco/build` (override with `MINCO_BUILD_DIR`); it degrades gracefully if not built.

## Training

Run from the repository root (Hydra configs live in `diffusion_policy/config/`).

```bash
# LAGO Policy (LA-CFG + goal head — main model)
python train.py --config-name=train_goal_aware_robot_workspace \
    task.dataset_path=/abs/path/to/dataset.zarr.zip

# LA-CFG only (ablation, no goal head)
python train.py --config-name=train_diffusion_unet_CFG_robot_workspace \
    task.dataset_path=/abs/path/to/dataset.zarr.zip
```

**Key hyperparameters:** Tp=16, To=2, Ta=8, Tf=4; DDIM 100/8 (train/infer); AdamW lr=1e-4 cosine; batch=64; 350 epochs; cf_drop_prob=0.3; λ_goal=0.1.


## Data Format

Trains on the UMI replay-buffer format: a [Zarr](https://zarr.dev/) group packed as `.zarr.zip` (default `data/robot_dataset/dataset.zarr.zip`, override with `task.dataset_path=`).

| Field | Shape | Notes |
|---|---|---|
| `camera{0,1,2}_rgb` | `(3, 224, 224)` | 2 workspace cams + 1 wrist fisheye |
| `robot0_eef_pos` / `robot0_eef_rot_axis_angle` / `robot0_gripper_width` | `(3,)` / `(6,)` / `(1,)` | EE pose (6D rotation) + gripper |
| action / goal | `(10,)` | `[pos(3), rot6d(6), gripper(1)]` |

Goals are 10-D and generated on-the-fly by `GoalAwareUmiDataset` from gripper-change skill points. To use your own data, package it into this UMI zarr format.

## Real-Robot Deployment

To run on a real robot, implement your own camera, arm, and gripper I/O in the three hardware hooks in [`eval.py`](eval.py), then run:

```bash
python eval.py
```

## Citation

```bibtex
@article{shi2026lago,
  title   = {LAGO Policy: Latency-Aware Asynchronous Diffusion Policies with
             Goal-Directed Collision-Free Planning for Smooth Manipulation},
  author  = {Shi, Guowei and Xie, Xupeng and Luo, Yiming and Guo, Jian and Ma, Jun and Zhou, Boyu},
  journal = {arXiv preprint arXiv:2606.17982},
  year    = {2026},
  url     = {https://lago-policy.github.io/}
}
```

## Acknowledgements

Built on [Diffusion Policy / UMI](https://github.com/real-stanford/universal_manipulation_interface) (policy backbone, replay-buffer format, training scaffolding), [EGO-Planner](https://github.com/ZJU-FAST-Lab/ego-planner) (collision-free B-spline planning), and [MINCO / GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER) (spatial-temporal optimization). Thanks to the authors for releasing their code.

## License

[MIT](LICENSE), inherited from UMI / Diffusion Policy. Please also respect the EGO-Planner and MINCO licenses.
