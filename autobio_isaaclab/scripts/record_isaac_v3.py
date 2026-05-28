#!/usr/bin/env python3
"""Render Isaac Sim static scene from de-instanced Newton USD — v3.

Fixes: hides meshes with world bounds > 10m (Newton converter artifacts
that cause camera frustum issues).

Usage:
    cd autobio-review/AutoBio/autobio_isaaclab
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    /home/robot/miniconda3/envs/isaac5/bin/python scripts/record_isaac_v3.py [--task TASK]
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
HUGE_THRESHOLD = 10.0  # meters

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
        from pxr import Usd, UsdGeom

        world = World(physics_dt=1.0/120.0, rendering_dt=1.0/FPS)
        add_reference_to_stage(usd_path=str(usd_path.resolve()),
                               prim_path=f"/World/{task_name}")
        world.reset()

        # Hide meshes with world bounds > threshold (Newton converter artifacts)
        stage = omni.usd.get_context().get_stage()
        hidden = 0
        for p in stage.Traverse():
            if p.GetTypeName() == "Mesh":
                try:
                    bbox = UsdGeom.Imageable(p).ComputeWorldBound(
                        Usd.TimeCode.Default(), "default", "default")
                    if bbox:
                        rng = bbox.GetRange()
                        sx = float(rng.GetMax()[0] - rng.GetMin()[0])
                        sy = float(rng.GetMax()[1] - rng.GetMin()[1])
                        sz = float(rng.GetMax()[2] - rng.GetMin()[2])
                        if sx > HUGE_THRESHOLD or sy > HUGE_THRESHOLD or sz > HUGE_THRESHOLD:
                            UsdGeom.Imageable(p).GetVisibilityAttr().Set("invisible")
                            hidden += 1
                except Exception:
                    pass
        status["hidden_meshes"] = hidden

        # Setup rendering
        rep.create.light(light_type="Distant", intensity=5000, rotation=(45, 45, 0))
        rep.create.light(light_type="Dome", intensity=1000)
        cam = rep.create.camera(position=(3.0, 3.0, 2.0), look_at=(0, 0, 0.5))
        rp = rep.create.render_product(cam, resolution=(WIDTH, HEIGHT))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([rp])

        # Warm up
        for _ in range(20):
            world.step(render=True)

        # Check first frame
        data = rgb_annot.get_data()
        if data is not None and data.size > 4:
            first_mean = float(np.mean(data[:, :, :3]))
        else:
            first_mean = 0.0

        # Capture frames
        frames = []
        for _ in range(STATIC_FRAMES):
            world.step(render=True)
            data = rgb_annot.get_data()
            if data is not None and data.size > 4:
                frames.append(data[:, :, :3].copy())

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
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    python = sys.executable
    script = str(Path(__file__).resolve())

    if args.task:
        if args.task not in TASKS:
            print(f"Unknown task: {args.task}")
            sys.exit(1)
        r = record_static(args.task)
        print(f"{r['task']}: ok={r['ok']} mean={r.get('first_mean','?')} "
              f"size={r.get('size_mb','?')}MB hidden={r.get('hidden_meshes',0)}")
    else:
        results = []
        for task_name in TASKS:
            print(f"Rendering {task_name}...")
            result_file = DEMO_DIR / f"{task_name}_isaac_result.json"
            env = os.environ.copy()
            env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
            proc = subprocess.run(
                [python, script, "--task", task_name],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result_file.exists():
                r = json.loads(result_file.read_text())
            else:
                r = {"task": task_name, "ok": False, "error": "no result file"}
            results.append(r)
            print(f"  {r['task']}: ok={r['ok']} mean={r.get('first_mean','?')} "
                  f"size={r.get('size_mb','?')}MB hidden={r.get('hidden_meshes',0)}")

        ok = [r for r in results if r["ok"]]
        fail = [r for r in results if not r["ok"]]
        summary_path = DEMO_DIR / "isaac_v3_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        print(f"\nSUMMARY: {len(ok)}/{len(results)} OK")
        for r in ok:
            print(f"  {r['task']}: {r.get('frames','?')} frames, "
                  f"{r.get('size_mb','?')} MB, mean={r.get('first_mean','?')}")
        for r in fail:
            print(f"  FAIL {r['task']}: {r.get('error','?')[:100]}")


if __name__ == "__main__":
    main()
