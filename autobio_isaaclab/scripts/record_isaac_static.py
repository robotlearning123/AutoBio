#!/usr/bin/env python3
"""Render Isaac Sim static scene from Newton USD assets.

Phase 1: Just load and render static meshes to verify they're visible.
Phase 2: Add MuJoCo kinematic animation.

Usage:
    cd autobio-review/AutoBio/autobio_isaaclab
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    conda run -n isaac5 --cwd . python scripts/record_isaac_static.py --task pickup
"""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

import json
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

BASE = Path(__file__).parent.parent
USD_ROOT = BASE / "usd_assets"
DEMO_DIR = BASE / "demos"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
FPS = 24
STATIC_FRAMES = 48  # 2 seconds of static video

TASKS = {
    "pickup": "pickup",
    "insert": "insert",
    "thermal_cycler_open": "mani_thermal_cycler",
    "thermal_cycler_close": "mani_thermal_cycler",
    "pipette": "mani_pipette",
    "screw_loose": "lab_screw_all",
    "screw_tighten": "lab_screw_all",
    "thermal_mixer": "mani_thermal_mixer",
    "vortex_mixer": "vortex_mixer",
    "insert_centrifuge_5430": "insert_centrifuge_5430",
}


def find_usd_path(scene_name: str) -> Path:
    scene_dir = USD_ROOT / scene_name / scene_name
    for ext in (".usda", ".usd"):
        p = scene_dir / f"{scene_name}{ext}"
        if p.exists():
            return p
    return scene_dir / f"{scene_name}.usda"


def record_static(task_name: str) -> dict:
    scene_name = TASKS[task_name]
    usd_path = find_usd_path(scene_name)
    output_path = DEMO_DIR / f"{task_name}_isaac.mp4"
    result_path = DEMO_DIR / f"{task_name}_isaac_result.json"
    status = {"task": task_name, "scene": scene_name, "ok": False, "error": None}

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
        import omni.usd
        from pxr import Usd

        world = World(physics_dt=1.0/120.0, rendering_dt=1.0/FPS)

        abs_usd = str(usd_path.resolve())
        prim_path = f"/World/{task_name}"
        add_reference_to_stage(usd_path=abs_usd, prim_path=prim_path)
        world.reset()

        stage = omni.usd.get_context().get_stage()
        prim_count = 0
        for _ in stage.Traverse():
            prim_count += 1

        cam = rep.create.camera(position=(2.0, 2.0, 1.8), look_at=(0, 0, 0.5))
        rp = rep.create.render_product(cam, resolution=(WIDTH, HEIGHT))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([rp])

        for _ in range(10):
            world.step(render=True)

        frames = []
        for _ in range(STATIC_FRAMES):
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
            status["prims"] = prim_count
            mean_val = float(np.mean(frames[0]))
            status["mean_pixel"] = round(mean_val, 1)
        else:
            status["error"] = "no_frames"

    except Exception as e:
        import traceback
        traceback.print_exc()
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
    if args.task not in TASKS:
        print(f"Unknown task: {args.task}")
        sys.exit(1)
    result = record_static(args.task)
    with open(str(DEMO_DIR / f"{args.task}_isaac_result.json")) as f:
        pass  # just verify it was written


if __name__ == "__main__":
    main()
