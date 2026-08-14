"""Kinematically replay a converted GR3Mini V2.1.1 tracking motion in Isaac Sim.

The script writes each reference frame directly to the articulation state.  It
is intended to verify the motion conversion, joint order, and visual quality;
it does not run a policy or physics simulation.

Example:

    python scripts/utils/replay_gr3mini_motion.py \
        --motion-file data/gr3mini/custom/Extended_3_stageii_from_g1_gr3mini_v211.npz

Use ``--headless --max-frames 10`` for a short non-interactive smoke test.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTION_FILE = (
    PROJECT_ROOT / "data/gr3mini/custom/Extended_3_stageii_from_g1_gr3mini_v211.npz"
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--motion-file",
    type=Path,
    default=DEFAULT_MOTION_FILE,
    help="Tracking-ready GR3Mini motion NPZ to replay.",
)
parser.add_argument(
    "--start-frame",
    type=int,
    default=0,
    help="Frame at which to start playback (default: 0).",
)
parser.add_argument(
    "--max-frames",
    type=int,
    default=0,
    help="Exit after this many rendered frames; 0 loops until the simulator closes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Isaac Sim must be launched before importing simulator-dependent modules.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.gr3mini import (
    GR3MINI_V211_CFG,
    GR3MINI_V211_ISAAC_JOINT_NAMES,
)
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayGr3MiniSceneCfg(InteractiveSceneCfg):
    """Scene containing one GR3Mini and a flat visual ground plane."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )
    robot: ArticulationCfg = GR3MINI_V211_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _validate_motion_file(path: Path) -> tuple[float, int]:
    """Check the conversion contract before launching the replay scene."""

    if not path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "joint_names",
            "body_names",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing required fields {sorted(missing)}")

        frame_count, joint_count = data["joint_pos"].shape
        if frame_count < 2 or joint_count != len(GR3MINI_V211_ISAAC_JOINT_NAMES):
            raise ValueError(
                f"{path}: expected at least 2 frames of "
                f"{len(GR3MINI_V211_ISAAC_JOINT_NAMES)}-DoF data, got "
                f"{data['joint_pos'].shape}."
            )
        if data["joint_vel"].shape != data["joint_pos"].shape:
            raise ValueError(f"{path}: joint_vel shape does not match joint_pos.")
        if data["body_pos_w"].shape != (frame_count, 24, 3):
            raise ValueError(
                f"{path}: expected body_pos_w shape {(frame_count, 24, 3)}, got "
                f"{data['body_pos_w'].shape}."
            )

        joint_names = tuple(str(name) for name in data["joint_names"].tolist())
        if joint_names != tuple(GR3MINI_V211_ISAAC_JOINT_NAMES):
            raise ValueError(
                f"{path}: joint order does not match the GR3Mini Isaac Lab articulation."
            )
        body_names = tuple(str(name) for name in data["body_names"].tolist())
        if not body_names or body_names[0] != "base_link":
            raise ValueError(f"{path}: body_names[0] must be 'base_link'.")

        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{path}: invalid fps {fps}.")

    return fps, frame_count


def _run_replay(
    sim: SimulationContext,
    scene: InteractiveScene,
    motion_file: Path,
    start_frame: int,
    max_frames: int,
) -> None:
    robot: Articulation = scene["robot"]
    root_body_index = torch.tensor([0], dtype=torch.long, device=sim.device)
    motion = MotionLoader(str(motion_file), root_body_index, sim.device)
    sim_dt = sim.get_physics_dt()
    frame = start_frame % motion.time_step_total
    rendered_frames = 0

    print(
        f"[INFO] Replaying {motion_file} ({motion.time_step_total} frames at "
        f"{float(motion.fps.reshape(-1)[0]):g} Hz); close the Isaac Sim window to stop."
    )

    while simulation_app.is_running():
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = motion.body_pos_w[frame, 0] + scene.env_origins
        root_state[:, 3:7] = motion.body_quat_w[frame, 0]
        root_state[:, 7:10] = motion.body_lin_vel_w[frame, 0]
        root_state[:, 10:13] = motion.body_ang_vel_w[frame, 0]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(
            motion.joint_pos[frame].unsqueeze(0),
            motion.joint_vel[frame].unsqueeze(0),
        )
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        look_at = root_state[0, :3].cpu().numpy()
        sim.set_camera_view(look_at + np.array([2.0, 2.0, 0.8]), look_at)

        frame = (frame + 1) % motion.time_step_total
        rendered_frames += 1
        if max_frames and rendered_frames >= max_frames:
            break


def main() -> None:
    motion_file = args_cli.motion_file.expanduser().resolve()
    fps, frame_count = _validate_motion_file(motion_file)
    if args_cli.max_frames < 0:
        raise ValueError("--max-frames must be non-negative.")

    sim = SimulationContext(
        sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / fps)
    )
    scene = InteractiveScene(ReplayGr3MiniSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print(
        f"[INFO] Motion contract validated: {frame_count} frames, 23 DoF, "
        f"24 bodies, {fps:g} Hz."
    )
    _run_replay(sim, scene, motion_file, args_cli.start_frame, args_cli.max_frames)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
