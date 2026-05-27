#!/usr/bin/env python3
"""Training entry point for AutoBio Isaac Lab environments.

Usage:
    cd /mnt/storage/isaacsim-6.0-official/
    ./isaaclab.sh -p /path/to/autobio_isaaclab/scripts/train.py --task Isaac-AutoBio-Pickup-Direct-v0

    # With custom parameters:
    ./isaaclab.sh -p scripts/train.py \\
        --task Isaac-AutoBio-Pickup-Direct-v0 \\
        --num_envs 128 \\
        --headless \\
        --max_iterations 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add package to path
_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Train AutoBio Isaac Lab environment")
    parser.add_argument("--task", type=str, default="Isaac-AutoBio-Pickup-Direct-v0",
                        help="Gymnasium environment ID")
    parser.add_argument("--num_envs", type=int, default=64,
                        help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("--max_iterations", type=int, default=500,
                        help="Maximum training iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import Isaac Lab app launcher (must be first)
    from isaaclab.app import AppLauncher

    # Parse app launcher args
    app_launcher_args = argparse.Namespace(
        headless=args.headless,
        device="cuda:0",
    )
    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app

    # Now import the rest
    import gymnasium as gym
    import torch

    # Register AutoBio environments
    import autobio_isaaclab

    # Create environment
    env = gym.make(args.task, num_envs=args.num_envs, cfg={"seed": args.seed})

    print(f"Created environment: {args.task}")
    print(f"  Num envs: {args.num_envs}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    # Simple random policy loop for testing
    obs, info = env.reset()
    print(f"  Initial obs shape: {obs['policy'].shape}")

    for i in range(args.max_iterations):
        actions = torch.randn(args.num_envs, env.action_space.shape[-1], device=obs["policy"].device)
        obs, rewards, terminated, truncated, info = env.step(actions)

        if i % 100 == 0:
            print(f"  Step {i}: reward_mean={rewards.mean().item():.4f}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
