#!/usr/bin/env python3
"""Evaluation script matching AutoBio's evaluator interface.

Usage:
    cd /mnt/storage/isaacsim-6.0-official/
    ./isaaclab.sh -p /path/to/autobio_isaaclab/scripts/evaluate.py \\
        --task Isaac-AutoBio-Pickup-Direct-v0 \\
        --num_episodes 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AutoBio environment")
    parser.add_argument("--task", type=str, default="Isaac-AutoBio-Pickup-Direct-v0")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--save", type=str, default=None, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(argparse.Namespace(headless=args.headless, device="cuda:0"))
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import autobio_isaaclab

    env = gym.make(args.task, num_envs=args.num_envs, cfg={"seed": args.seed})

    master_rng = np.random.default_rng(args.seed)
    results = []

    for ep in range(args.num_episodes):
        seed = int(master_rng.integers(0, 2**32 - 1))
        obs, info = env.reset(seed=seed)

        done = False
        while not done:
            # Random policy for now - replace with trained policy
            actions = torch.randn(args.num_envs, env.action_space.shape[-1], device=obs["policy"].device)
            obs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated.any() or truncated.any()

        success = rewards.sum().item() > 0
        results.append(float(success))
        print(f"Episode {ep}: {'SUCCESS' if success else 'FAIL'}")

    env.close()
    simulation_app.close()

    success_rate = sum(results) / len(results)
    print(f"\nSuccess rate: {success_rate:.2%} ({int(sum(results))}/{len(results)})")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
