#!/usr/bin/env python3
"""Record Isaac Sim demo videos for AutoBio tasks.

Single app instance, records one scene at a time. Each task needs its own
process (SimulationApp singleton).

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    conda run -n isaac5 --cwd autobio_isaaclab python scripts/record_isaac_videos.py --task pickup
"""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

import json
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

USD_ROOT = Path(__file__).parent.parent / "usd_assets_physx"
DEMO_DIR = Path(__file__).parent.parent / "demos"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
FPS = 24
NUM_STEPS = 500

SCENES = {
    "pickup": "pickup.usd",
    "insert": "insert.usd",
    "thermal_cycler_open": "mani_thermal_cycler.usd",
    "thermal_cycler_close": "mani_thermal_cycler.usd",
    "pipette": "mani_pipette.usd",
    "screw_loose": "lab_screw_all.usd",
    "screw_tighten": "lab_screw_all.usd",
    "thermal_mixer": "mani_thermal_mixer.usd",
    "vortex_mixer": "vortex_mixer.usd",
    "insert_centrifuge_5430": "insert_centrifuge_5430.usd",
}


def record_scene(task_name: str) -> dict:
    """Record one scene."""
    usd_file = SCENES[task_name]
    usd_path = USD_ROOT / usd_file
    output_path = DEMO_DIR / f"{task_name}_isaac.mp4"
    result_path = DEMO_DIR / f"{task_name}_isaac_result.json"
    status = {"task": task_name, "ok": False, "error": None}

    if not usd_path.exists():
        status["error"] = f"USD not found: {usd_path}"
        result_path.write_text(json.dumps(status))
        return status

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    try:
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import omni.replicator.core as rep

        world = World(physics_dt=1.0/120.0, rendering_dt=1.0/FPS)

        prim_path = f"/World/{task_name}"
        add_reference_to_stage(usd_path=str(usd_path), prim_path=prim_path)
        world.reset()

        # Camera
        cam = rep.create.camera(position=(1.5, 1.5, 1.5), look_at=(0, 0, 0.5))
        rp = rep.create.render_product(cam, resolution=(WIDTH, HEIGHT))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([rp])

        # Warm up
        for _ in range(5):
            world.step(render=True)

        frames = []
        for step_i in range(NUM_STEPS):
            world.step(render=True)
            data = rgb_annot.get_data()
            if data is not None and data.size > 0:
                frame = data[:, :, :3].copy()
                frames.append(frame)

        if frames:
            imageio.mimwrite(str(output_path), frames, fps=FPS, codec="libx264",
                           output_params=["-crf", "18"])
            size_mb = output_path.stat().st_size / (1024 * 1024)
            status["ok"] = True
            status["frames"] = len(frames)
            status["size_mb"] = round(size_mb, 2)
        else:
            status["error"] = "no_frames"

    except Exception as e:
        status["error"] = str(e)[:300]
    finally:
        app.close()

    result_path.write_text(json.dumps(status))
    return status


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    result = record_scene(args.task)


if __name__ == "__main__":
    main()
