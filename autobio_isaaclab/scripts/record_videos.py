#!/usr/bin/env python3
"""Record demo videos for all AutoBio Isaac Lab tasks.

Uses gym.wrappers.RecordVideo with render_mode="rgb_array" (official Isaac Lab pattern).
Runs each task for N steps with random actions, captures viewport frames, writes mp4.

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \\
    conda run -n isaac5 python scripts/record_videos.py --headless
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

parser = argparse.ArgumentParser(description="Record AutoBio demo videos")
parser.add_argument("--num_steps", type=int, default=80)
parser.add_argument("--fps", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym

import autobio_isaaclab.envs.tasks


ALL_TASKS = [
    ("Isaac-AutoBio-Pickup-Direct-v0", "pickup"),
    ("Isaac-AutoBio-Insert-Direct-v0", "insert"),
    ("Isaac-AutoBio-ThermalCyclerOpen-Direct-v0", "thermal_cycler_open"),
    ("Isaac-AutoBio-ThermalCyclerClose-Direct-v0", "thermal_cycler_close"),
    ("Isaac-AutoBio-Pipette-Direct-v0", "pipette"),
    ("Isaac-AutoBio-ScrewLoose-Direct-v0", "screw_loose"),
    ("Isaac-AutoBio-ScrewTighten-Direct-v0", "screw_tighten"),
    ("Isaac-AutoBio-Centrifuge5430-Direct-v0", "centrifuge_5430"),
    ("Isaac-AutoBio-Centrifuge5910-Direct-v0", "centrifuge_5910"),
    ("Isaac-AutoBio-CentrifugeMini-Direct-v0", "centrifuge_mini"),
    ("Isaac-AutoBio-ThermalMixer-Direct-v0", "thermal_mixer"),
    ("Isaac-AutoBio-VortexMixer-Direct-v0", "vortex_mixer"),
]

OUTPUT_DIR = ROOT / "demos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def record_task(task_id: str, short_name: str) -> dict:
    """Record a single task using RecordVideo wrapper."""
    video_dir = OUTPUT_DIR / short_name
    video_dir.mkdir(parents=True, exist_ok=True)
    status = {"task": short_name, "ok": False, "error": None}

    print(f"\n{'='*60}")
    print(f"Recording: {task_id}")
    print(f"{'='*60}")

    env = None
    try:
        env = gym.make(task_id, num_envs=1, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_dir=str(video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.num_steps,
            disable_logger=True,
            name_prefix=short_name,
        )

        obs, info = env.reset()
        print(f"  reset OK, obs shape: {obs['policy'].shape}")

        for step_i in range(args.num_steps):
            action = env.action_space.sample()
            action_tensor = torch.tensor(action, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            obs, reward, terminated, truncated, info = env.step(action_tensor)

            if step_i % 20 == 0:
                print(f"  step {step_i}/{args.num_steps}, reward: {reward.item():.3f}")

            if terminated.any() or truncated.any():
                print(f"  episode ended at step {step_i}")
                break

        env.close()

        # Check if video was written
        videos = list(video_dir.glob("*.mp4"))
        if videos:
            latest = max(videos, key=lambda p: p.stat().st_mtime)
            # Rename to canonical name
            final = OUTPUT_DIR / f"{short_name}.mp4"
            if final.exists():
                final.unlink()
            latest.rename(final)
            # Remove temp dir if empty
            try:
                video_dir.rmdir()
            except OSError:
                pass
            size_mb = final.stat().st_size / (1024 * 1024)
            print(f"  Wrote {final.name} ({size_mb:.1f} MB)")
            status["ok"] = True
            status["size_mb"] = round(size_mb, 2)
            status["path"] = str(final)
        else:
            print(f"  WARNING: RecordVideo produced no mp4")
            status["error"] = "no_video_output"

    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        status["error"] = str(e)[:300]
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return status


def record_task_manual(task_id: str, short_name: str) -> dict:
    """Fallback: manual frame capture if RecordVideo doesn't work with DirectRLEnv."""
    import imageio.v2 as imageio

    status = {"task": short_name, "ok": False, "error": None}
    print(f"\n{'='*60}")
    print(f"Recording (manual): {task_id}")
    print(f"{'='*60}")

    env = None
    frames = []
    try:
        env = gym.make(task_id, num_envs=1, render_mode="rgb_array")
        obs, info = env.reset()
        print(f"  reset OK, obs shape: {obs['policy'].shape}")

        for step_i in range(args.num_steps):
            action = env.action_space.sample()
            action_tensor = torch.tensor(action, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            obs, reward, terminated, truncated, info = env.step(action_tensor)

            frame = env.render()
            if frame is not None:
                frames.append(np.array(frame))

            if step_i % 20 == 0:
                print(f"  step {step_i}/{args.num_steps}, frames: {len(frames)}")

            if terminated.any() or truncated.any():
                break

        env.close()

        if frames:
            output_path = OUTPUT_DIR / f"{short_name}.mp4"
            imageio.mimwrite(str(output_path), frames, fps=args.fps, codec="libx264",
                           output_params=["-crf", "20", "-pix_fmt", "yuv420p"])
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Wrote {len(frames)} frames -> {output_path.name} ({size_mb:.1f} MB)")
            status["ok"] = True
            status["frames"] = len(frames)
            status["size_mb"] = round(size_mb, 2)
            status["path"] = str(output_path)
        else:
            print(f"  WARNING: No frames captured")
            status["error"] = "no_frames"

    except Exception as e:
        print(f"  FAIL: {e}")
        traceback.print_exc()
        status["error"] = str(e)[:300]
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return status


def main():
    print(f"AutoBio Isaac Lab Video Recorder")
    print(f"  Tasks: {len(ALL_TASKS)}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Output: {OUTPUT_DIR}")

    results = []
    for task_id, short_name in ALL_TASKS:
        # Try RecordVideo first, fall back to manual capture
        r = record_task(task_id, short_name)
        if not r["ok"]:
            print(f"  RecordVideo failed, trying manual capture...")
            r = record_task_manual(task_id, short_name)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print(f"  OK: {len(ok)}/{len(results)}")
    for r in ok:
        print(f"    {r['task']}: {r.get('size_mb', '?')} MB")
    for r in fail:
        print(f"    FAIL {r['task']}: {r.get('error', '?')[:100]}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "ok": len(ok),
        "fail": len(fail),
        "results": results,
    }
    with open(OUTPUT_DIR / "recording_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    simulation_app.close()


if __name__ == "__main__":
    main()
