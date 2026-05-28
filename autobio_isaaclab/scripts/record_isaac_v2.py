#!/usr/bin/env python3
"""Render Isaac Sim static scene from de-instanced Newton USD — v2.

Usage:
    cd autobio-review/AutoBio/autobio_isaaclab
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    /home/robot/miniconda3/envs/isaac5/bin/python scripts/record_isaac_v2.py --task insert
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
STATIC_FRAMES = 48

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


def find_deinst_usd(scene_name: str) -> Path:
    scene_dir = USD_ROOT / scene_name / scene_name
    deinst = scene_dir / f"{scene_name}_deinst.usda"
    if deinst.exists():
        return deinst
    for ext in (".usda", ".usd"):
        p = scene_dir / f"{scene_name}{ext}"
        if p.exists():
            return p
    return deinst


def record_static(task_name: str) -> dict:
    scene_name = TASKS[task_name]
    usd_path = find_deinst_usd(scene_name)
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

        # Critical: World() MUST be created before add_reference_to_stage
        world = World(physics_dt=1.0/120.0, rendering_dt=1.0/FPS)

        # Load USD scene first
        abs_usd = str(usd_path.resolve())
        prim_path = f"/World/{task_name}"
        add_reference_to_stage(usd_path=abs_usd, prim_path=prim_path)
        world.reset()

        # Setup rendering AFTER scene is loaded
        rep.create.light(light_type="Distant", intensity=5000, rotation=(45, 45, 0))
        rep.create.light(light_type="Dome", intensity=1000)
        cam = rep.create.camera(position=(3.0, 3.0, 2.0), look_at=(0, 0, 0.5))
        rp = rep.create.render_product(cam, resolution=(640, 480))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([rp])

        # Warm up renderer with scene
        for _ in range(20):
            world.step(render=True)

        # Check first frame
        data = rgb_annot.get_data()
        if data is not None and data.size > 4:
            first_mean = float(np.mean(data[:, :, :3]))
        else:
            first_mean = 0.0

        # If still black, try adjusting camera
        if first_mean < 1.0:
            # Scene might have unusual bounds — try a farther camera
            rep.create.camera(position=(5.0, 5.0, 3.0), look_at=(0, 0, 0.5))
            world.reset()
            for _ in range(15):
                world.step(render=True)
            data = rgb_annot.get_data()
            if data is not None and data.size > 4:
                first_mean = float(np.mean(data[:, :, :3]))

        frames = []
        for _ in range(STATIC_FRAMES):
            world.step(render=True)
            data = rgb_annot.get_data()
            if data is not None and data.size > 4:
                frame = data[:, :, :3].copy()
                frames.append(frame)

        if frames:
            imageio.mimwrite(str(output_path), frames, fps=FPS, codec="libx264",
                           output_params=["-crf", "18"])
            size_mb = output_path.stat().st_size / (1024 * 1024)
            status["ok"] = True
            status["frames"] = len(frames)
            status["size_mb"] = round(size_mb, 2)
            status["first_mean"] = round(first_mean, 1)
        else:
            status["error"] = "no_frames"
            status["first_mean"] = round(first_mean, 1)

    except Exception as e:
        import traceback
        traceback.print_exc()
        status["error"] = str(e)[:300]
    finally:
        result_path.write_text(json.dumps(status))
        app.close()

    return status


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    if args.task:
        if args.task not in TASKS:
            print(f"Unknown task: {args.task}")
            sys.exit(1)
        r = record_static(args.task)
        print(f"{r['task']}: ok={r['ok']} mean={r.get('first_mean','?')} size={r.get('size_mb','?')}MB")
    else:
        results = []
        for task_name in TASKS:
            r = record_static(task_name)
            results.append(r)
            print(f"{r['task']}: ok={r['ok']} mean={r.get('first_mean','?')} size={r.get('size_mb','?')}MB")
        ok = [r for r in results if r["ok"]]
        fail = [r for r in results if not r["ok"]]
        summary_path = DEMO_DIR / "isaac_v2_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        print(f"\nSUMMARY: {len(ok)}/{len(results)} OK")
        for r in ok:
            print(f"  {r['task']}: {r.get('frames','?')} frames, {r.get('size_mb','?')} MB, mean={r.get('first_mean','?')}")
        for r in fail:
            print(f"  FAIL {r['task']}: {r.get('error','?')[:100]}")


if __name__ == "__main__":
    main()
