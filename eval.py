#!/usr/bin/env python3
"""LAGO Policy -- real-robot deployment reference.

Implement the 3 hardware hooks below for your own camera/arm/gripper, then run
``python eval.py``.

This is a single, self-contained *reference* script: it loads a trained LAGO
checkpoint, runs the synchronous observe -> predict -> (optionally plan) ->
execute control loop, and converts between the robot and the policy. It is meant
to be read and adapted -- it will not run end-to-end until the three
``NotImplementedError`` hooks are filled in for your hardware.

Note: the policy is trained latency-aware (LA-CFG conditions on future actions),
so for a true asynchronous deployment run ``policy.predict_action`` in a
background thread while the arm keeps executing the previous chunk; the loop
below is kept synchronous for clarity.
"""
import os
import sys
import time
import argparse

import numpy as np

# Make the release importable (diffusion_policy / planning) when run from
# anywhere. eval.py lives at the repo root (next to train.py), so ROOT_DIR is
# this file's own directory.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2
import dill
import hydra
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from diffusion_policy.policy.goal_aware_diffusion_policy import GoalAwareDiffusionPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
# Pose helpers MUST match training: rot6d = first 2 ROWS of the rotation matrix.
from diffusion_policy.common.pose_util import pose_to_mat, mat_to_pose10d, rot6d_to_mat

# Goal-directed collision-free planning. These degrade gracefully: if the C++
# extensions (ego_planner_cpp / MINCO joint_traj_opt) are not built, the
# *_AVAILABLE flags are False and the loop falls back to raw action chunks.
from planning import (
    EGO_PLANNER_AVAILABLE,
    EgoPlannerManager,
    GoalDirectedModeSelector,
    goal_to_pose,
    interpolate_pose_by_time,
)
from planning.ego_planner_manager import SAMPLING_INTERVAL
from planning.minco_trajectory_optimizer import MINCO_AVAILABLE, optimize_diffusion_trajectory

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ----- Config (README only shows `python eval.py`, so defaults must be sane) -----
CHECKPOINT = "checkpoint.ckpt"   # default ckpt path; override with -i/--checkpoint
DEVICE = "cuda"
CONTROL_HZ = 20.0                # control / policy frequency (Hz)
MAX_DURATION = 120.0             # stop after this many seconds

# LAGO goal-directed planning. Set ENABLE_GOAL_PLANNING=False for the LA-CFG-only
# behaviour (always execute the raw action chunk, ignore the predicted goal).
ENABLE_GOAL_PLANNING = True
USE_EGO_PLANNER = False          # collision-free B-spline in NAV mode (needs ego_planner_cpp)
USE_MINCO = False                # spatial-temporal smoothing (needs MINCO joint_traj_opt)


# ----- Policy loading (build from the saved cfg, then load the EMA weights) -----
def load_policy(checkpoint_path: str, device: str):
    """Build the policy from the checkpoint's saved cfg and load its weights.

    Returns ``(policy, cfg)``. EMA weights are preferred (smoother). The normalizer
    is a submodule of the policy, so ``load_state_dict`` also restores its buffers
    -- no dataset or separate ``set_normalizer`` call is needed at deploy time.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = OmegaConf.create(payload["cfg"])

    policy: GoalAwareDiffusionPolicy = hydra.utils.instantiate(cfg.policy)
    state_dict = payload.get("ema_model", payload.get("policy"))
    policy.load_state_dict(state_dict)

    policy.eval().to(device)
    policy.reset()  # clears the LA-CFG future-action condition for a new episode
    return policy, cfg


# ===== THREE HARDWARE HOOKS -- implement these for YOUR robot/cameras/gripper =====
def get_camera_images() -> dict:
    """HOOK 1/3: return the latest RGB frame from each camera.

    Returns a dict ``{'camera{i}_rgb': HxWx3 array}`` (HWC, RGB order, uint8
    [0,255] or float32 [0,1]). The keys/count/order must match the policy's
    ``shape_meta`` (the bundled config expects camera0_rgb, camera1_rgb,
    camera2_rgb). Resizing to the model resolution is handled downstream.
    """
    # TODO: implement for YOUR hardware (RealSense / UVC / fisheye / ...).
    raise NotImplementedError("get_camera_images(): read your cameras here")


def get_robot_state():
    """HOOK 2/3: return the current end-effector pose and gripper width.

    Returns ``(eef_pos, eef_rot_axis_angle, gripper_width)`` where
    ``eef_pos`` is (3,) [x,y,z] in meters (world frame), ``eef_rot_axis_angle``
    is (3,) axis-angle (rotvec, world frame), and ``gripper_width`` is a float
    (meters).
    """
    # TODO: implement for YOUR hardware (arm FK / TCP pose + gripper feedback).
    raise NotImplementedError("get_robot_state(): read your arm + gripper here")


def send_robot_action(action: np.ndarray) -> None:
    """HOOK 3/3: command the arm + gripper toward one 7-D target.

    ``action`` is ``[x, y, z, rx, ry, rz, gripper_width]`` -- absolute EE
    position (m) + axis-angle orientation (rotvec) + gripper width (m), world
    frame. Drive your arm controller and gripper toward this target.
    """
    # TODO: implement for YOUR hardware (arm waypoint command + gripper command).
    raise NotImplementedError("send_robot_action(): drive your arm + gripper here")


# ----- Minimal obs/action conversion (self-contained; matches training) -----
def build_model_obs(obs_history: list, shape_meta: dict, obs_pose_repr: str) -> dict:
    """Convert a short history of raw observations into the policy obs dict.

    ``obs_history`` is a list of length ``n_obs_steps`` (oldest -> newest) of
    ``{'images': {camera{i}_rgb: HxWx3}, 'eef_pos': (3,), 'eef_rot_aa': (3,),
    'gripper': float}``. Returns float32 arrays keyed by the policy's obs names
    (images TCHW in [0,1]; pose as rot6d in ``obs_pose_repr``, as in training).
    """
    obs_meta = shape_meta["obs"]
    out = {}

    # --- cameras: stack frames, resize to model resolution, normalize, THWC->TCHW
    for key, attr in obs_meta.items():
        if attr.get("type", "low_dim") != "rgb":
            continue
        _, ho, wo = attr["shape"]
        frames = []
        for o in obs_history:
            img = o["images"][key]
            if img.shape[:2] != (ho, wo):
                img = cv2.resize(img, (wo, ho), interpolation=cv2.INTER_LINEAR)
            frames.append(img)
        imgs = np.stack(frames).astype(np.float32)
        if imgs.max() > 1.0:                  # uint8 [0,255] -> [0,1]
            imgs = imgs / 255.0
        out[key] = np.moveaxis(imgs, -1, 1)   # (T,H,W,3) -> (T,3,H,W)

    # --- pose: convert to the same representation used at training time ---
    pos = np.stack([o["eef_pos"] for o in obs_history]).astype(np.float64)        # (T,3)
    rot_aa = np.stack([o["eef_rot_aa"] for o in obs_history]).astype(np.float64)  # (T,3)
    pose_mat = pose_to_mat(np.concatenate([pos, rot_aa], axis=-1))                # (T,4,4)
    obs_pose = mat_to_pose10d(convert_pose_mat_rep(
        pose_mat, base_pose_mat=pose_mat[-1], pose_rep=obs_pose_repr, backward=False))
    if "robot0_eef_pos" in obs_meta:
        out["robot0_eef_pos"] = obs_pose[:, :3].astype(np.float32)
    if "robot0_eef_rot_axis_angle" in obs_meta:
        out["robot0_eef_rot_axis_angle"] = obs_pose[:, 3:9].astype(np.float32)
    if "robot0_gripper_width" in obs_meta:
        out["robot0_gripper_width"] = np.array(
            [[o["gripper"]] for o in obs_history], dtype=np.float32)              # (T,1)

    return out


def action_to_command(action: np.ndarray, base_pose: np.ndarray, obs_pose_repr: str) -> np.ndarray:
    """Convert one 10-D policy action into a 7-D absolute robot command.

    ``action`` is ``[pos(3), rot6d(6), gripper(1)]``; ``base_pose`` is the last
    observation EE pose ``[x,y,z,rx,ry,rz]`` (axis-angle). Returns
    ``[x,y,z,rx,ry,rz,gripper]`` in the absolute world frame.
    """
    rot_mat = rot6d_to_mat(action[3:9])
    action_mat = np.eye(4)
    action_mat[:3, :3] = rot_mat
    action_mat[:3, 3] = action[:3]

    if obs_pose_repr == "relative":
        # Convert the relative (w.r.t. last obs frame) action back to absolute.
        world_mat = convert_pose_mat_rep(
            action_mat, base_pose_mat=pose_to_mat(base_pose),
            pose_rep="relative", backward=True)
        target_pos = world_mat[:3, 3]
        target_rot = R.from_matrix(world_mat[:3, :3]).as_rotvec()
    else:  # 'abs': action is already an absolute pose
        target_pos = action[:3]
        target_rot = R.from_matrix(rot_mat).as_rotvec()

    gripper = action[9] if len(action) > 9 else 0.08
    return np.array([*target_pos, *target_rot, gripper])


def predict(policy, model_obs: dict, device: str):
    """Run one inference. Returns ``(action_chunk (Ta,10), goal (10,) or None)``.

    ``policy.predict_action`` carries the LA-CFG future-action condition
    internally across successive calls (reset per episode via ``policy.reset()``).
    """
    obs_t = dict_apply(model_obs,
        lambda x: torch.from_numpy(x).unsqueeze(0).to(device, dtype=torch.float32))
    with torch.no_grad():
        result = policy.predict_action(obs_t)
    action_chunk = result["action"][0].detach().cpu().numpy()
    goal = result["goal"][0].detach().cpu().numpy() if result.get("goal") is not None else None
    return action_chunk, goal


# ----- Control loop -----
def main():
    parser = argparse.ArgumentParser(description="LAGO Policy real-robot deployment reference")
    parser.add_argument("-i", "--checkpoint", default=CHECKPOINT, help="policy .ckpt path")
    parser.add_argument("--device", default=DEVICE, help="torch device")
    args = parser.parse_args()

    policy, cfg = load_policy(args.checkpoint, args.device)

    # Pull everything the loop needs from the saved config.
    shape_meta = OmegaConf.to_container(cfg.policy.shape_meta, resolve=True)
    n_obs_steps = int(cfg.policy.n_obs_steps)
    n_action_steps = int(cfg.policy.n_action_steps)
    obs_pose_repr = cfg.get("task", {}).get("pose_repr", {}).get("obs_pose_repr", "abs")
    dt = 1.0 / CONTROL_HZ

    # Optional goal-directed planning components (LAGO).
    mode_selector = GoalDirectedModeSelector() if ENABLE_GOAL_PLANNING else None
    ego_planner = None
    if ENABLE_GOAL_PLANNING and USE_EGO_PLANNER and EGO_PLANNER_AVAILABLE:
        ego_planner = EgoPlannerManager(obstacles=None)

    def sample_obs():
        images = get_camera_images()
        eef_pos, eef_rot_aa, gripper = get_robot_state()
        return {
            "images": images,
            "eef_pos": np.asarray(eef_pos, dtype=np.float64),
            "eef_rot_aa": np.asarray(eef_rot_aa, dtype=np.float64),
            "gripper": float(gripper),
        }

    print(f"LAGO deployment reference running (goal_planning={ENABLE_GOAL_PLANNING}, "
          f"ego={ego_planner is not None}, minco={USE_MINCO and MINCO_AVAILABLE}). Ctrl-C to stop.")
    obs_history = []
    t_start = time.time()
    while time.time() - t_start < MAX_DURATION:
        # Maintain a rolling history of the last n_obs_steps observations.
        obs_history.append(sample_obs())
        if len(obs_history) < n_obs_steps:
            continue
        obs_history = obs_history[-n_obs_steps:]
        cur = obs_history[-1]

        # observe -> predict
        model_obs = build_model_obs(obs_history, shape_meta, obs_pose_repr)
        action_chunk, goal = predict(policy, model_obs, args.device)
        base_pose = np.concatenate([cur["eef_pos"], cur["eef_rot_aa"]])

        # decide: navigate to the predicted goal (collision-free) or run the chunk
        commands = None
        exec_interval = dt
        if ENABLE_GOAL_PLANNING and goal is not None:
            gp = goal_to_pose(goal)
            # ee_velocity is 0.0 here (synchronous loop); estimate it from
            # feedback if you need the hysteresis exit condition to be precise.
            mode = mode_selector.decide(cur["eef_pos"], gp["position"], 0.0)
            if mode == "NAV":
                start_rot = R.from_rotvec(cur["eef_rot_aa"]).as_matrix()
                if ego_planner is not None:  # collision-free B-spline (EGO-Planner)
                    commands, _ = ego_planner.generate_commands_with_rotation(
                        cur["eef_pos"], gp["position"], start_rot, gp["rotation_matrix"],
                        cur["gripper"], gp["gripper"])
                if commands is None:         # fallback: linear + SLERP interpolation
                    dist = float(np.linalg.norm(cur["eef_pos"] - gp["position"]))
                    commands, _ = interpolate_pose_by_time(
                        cur["eef_pos"], start_rot, gp["position"], gp["rotation_matrix"],
                        cur["gripper"], gp["gripper"], duration=max(dist / 0.15, 1.0))
                commands = np.asarray(commands)
                exec_interval = SAMPLING_INTERVAL

        if commands is None:  # ACTION mode (planning off, no goal, or goal is close)
            commands = np.array([
                action_to_command(a, base_pose, obs_pose_repr)
                for a in action_chunk[:n_action_steps]])

        # optional MINCO spatial-temporal smoothing (200 Hz dense waypoints)
        if USE_MINCO and MINCO_AVAILABLE and len(commands) >= 2:
            timestamps = np.arange(len(commands)) * exec_interval + time.time()
            ok, dense, _ = optimize_diffusion_trajectory(commands, timestamps, verbose=False)
            if ok and dense is not None:
                commands = dense
                exec_interval = 0.005  # MINCO samples at 200 Hz

        # execute (synchronous: send each waypoint, then pace to its interval)
        for cmd in commands:
            send_robot_action(cmd)
            time.sleep(exec_interval)

    print("LAGO deployment reference: max duration reached, stopping.")


if __name__ == "__main__":
    main()
