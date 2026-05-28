#!/usr/bin/env python3
"""Batch smoke test for all AutoBio Isaac Lab environments.

Must be run via: isaaclab.sh -p scripts/test_all.py --headless
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Must launch Isaac Sim before any other imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import autobio_isaaclab.envs.tasks  # register envs

tasks = [
    "Isaac-AutoBio-Pickup-Direct-v0",
    "Isaac-AutoBio-Insert-Direct-v0",
    "Isaac-AutoBio-ThermalCyclerOpen-Direct-v0",
    "Isaac-AutoBio-ThermalCyclerClose-Direct-v0",
    "Isaac-AutoBio-Pipette-Direct-v0",
    "Isaac-AutoBio-ScrewLoose-Direct-v0",
    "Isaac-AutoBio-ScrewTighten-Direct-v0",
    "Isaac-AutoBio-Centrifuge5430-Direct-v0",
    "Isaac-AutoBio-ThermalMixer-Direct-v0",
    "Isaac-AutoBio-VortexMixer-Direct-v0",
    "Isaac-AutoBio-Centrifuge5910-Direct-v0",
    "Isaac-AutoBio-CentrifugeMini-Direct-v0",
]

passed = []
failed = []

for task in tasks:
    print(f"\n=== {task} ===")
    try:
        env = gym.make(task, num_envs=1, disable_env_checker=True)
        obs, info = env.reset()
        print(f"  reset OK, obs shape: {obs['policy'].shape}")
        action = torch.tensor(env.action_space.sample(), dtype=torch.float32, device="cuda:0")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  step OK, reward: {reward}")
        env.close()
        print(f"  PASS")
        passed.append(task)
    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        failed.append((task, str(e)))
        try:
            env.close()
        except Exception:
            pass

print(f"\n{'='*60}")
print(f"RESULTS: {len(passed)}/{len(tasks)} passed")
for t in passed:
    print(f"  PASS: {t}")
for t, e in failed:
    print(f"  FAIL: {t}: {e[:80]}")

simulation_app.close()
