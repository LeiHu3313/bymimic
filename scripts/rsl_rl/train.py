# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments per process; multiplied by world size with --distributed.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run synchronous multi-GPU training through torchrun.",
)
motion_group = parser.add_mutually_exclusive_group(required=True)
motion_group.add_argument(
    "--registry_name", type=str, help="The name of the WandB motion registry artifact."
)
motion_group.add_argument(
    "--motion_file", type=str, help="Path to a local tracking-ready motion NPZ file."
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import (
    MotionOnPolicyRunner as OnPolicyRunner,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlOnPolicyRunnerCfg,
):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    if (
        args_cli.distributed
        and args_cli.device is not None
        and "cpu" in args_cli.device.lower()
    ):
        raise ValueError(
            "Distributed training requires CUDA. Remove --device cpu or select a CUDA device."
        )
    if args_cli.distributed:
        # AppLauncher resolves LOCAL_RANK from torchrun and binds Isaac Sim to
        # this device. RSL-RL checks the same device string before initializing
        # its NCCL process group.
        local_rank = app_launcher.local_rank
        env_cfg.sim.device = f"cuda:{local_rank}"
        agent_cfg.device = f"cuda:{local_rank}"
        agent_cfg.seed += local_rank
        env_cfg.seed = agent_cfg.seed

    is_main_process = (
        not args_cli.distributed or getattr(app_launcher, "global_rank", 0) == 0
    )

    # Load either a local motion or a WandB registry artifact.
    registry_name = args_cli.registry_name
    if args_cli.motion_file is not None:
        motion_file = os.path.abspath(args_cli.motion_file)
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
        env_cfg.commands.motion.motion_file = motion_file
        print(f"[INFO] Using local motion file: {motion_file}")
    else:
        if (
            ":" not in registry_name
        ):  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(
            pathlib.Path(artifact.download()) / "motion.npz"
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if is_main_process:
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )
    # wrap for video recording
    if args_cli.video and not args_cli.distributed:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir if is_main_process else None,
        device=agent_cfg.device,
        registry_name=registry_name,
    )
    # write git state to logs
    if is_main_process:
        runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # Only the main process writes run metadata. The RSL-RL runner likewise
    # logs and checkpoints on global rank 0.
    if is_main_process:
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        with open(os.path.join(log_dir, "params", "env.pkl"), "wb") as file:
            pickle.dump(env_cfg, file)
        with open(os.path.join(log_dir, "params", "agent.pkl"), "wb") as file:
            pickle.dump(agent_cfg, file)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
