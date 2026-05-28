#!/usr/bin/env python3
"""Record a single AutoBio Isaac Lab task video.

Runs one task, captures frames via viewport render, writes mp4.
Designed to be called once per task (one sim context per process).

Usage:
    LD_PRELOAD=... conda run -n isaac5 python scripts/record_single.py \
        --task Isaac-AutoBio-Pickup-Direct-v0 --headless
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--name", type=str, default=None)
parser.add_argument("--num_steps", type=int, default=50)
parser.add_argument("--fps", type=int, default=24)
parser.add_argument("--output_dir", type=str, default=str(ROOT / "demos"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio

import autobio_isaaclab.envs.tasks

short_name = args.name or args.task.replace("Isaac-AutoBio-", "").replace("-Direct-v0", "")
output_path = Path(args.output_dir) / f"{short_name}.mp4"
output_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Recording: {args.task} -> {output_path.name}")

frames = []
try:
    env = gym.make(args.task, num_envs=1, render_mode="rgb_array")
    obs, info = env.reset()
    print(f"  reset OK, obs shape: {obs['policy'].shape}")

    for step_i in range(args.num_steps):
        action = env.action_space.sample()
        action_tensor = torch.tensor(action, dtype=torch.float32, device="cuda:0").unsqueeze(0)
        obs, reward, terminated, truncated, info = env.step(action_tensor)

        # Capture frame
        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame))

        if step_i % 10 == 0:
            print(f"  step {step_i}/{args.num_steps}, frames: {len(frames)}")

        if terminated.any() or truncated.any():
            print(f"  episode ended at step {step_i}")
            break

    env.close()

    if frames:
        imageio.mimwrite(str(output_path), frames, fps=args.fps, codec="libx264",
                       output_params=["-crf", "20", "-pix_fmt", "yuv420p"])
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Wrote {len(frames)} frames -> {output_path.name} ({size_mb:.1f} MB)")
        result = {"task": short_name, "ok": True, "frames": len(frames), "size_mb": round(size_mb, 2)}
    else:
        print(f"  WARNING: No frames captured")
        result = {"task": short_name, "ok": False, "error": "no_frames"}

except Exception as e:
    print(f"  FAIL: {e}")
    traceback.print_exc()
    result = {"task": short_name, "ok": False, "error": str(e)[:300]}

# Write per-task result JSON
result_path = Path(args.output_dir) / f"{short_name}_result.json"
with open(result_path, "w") as f:
    json.dump(result, f, indent=2)

simulation_app.close()
