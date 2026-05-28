#!/usr/bin/env python3
"""Record demo videos for all AutoBio tasks using MuJoCo renderer.

Hooks into Manager.step() to capture frames during expert execution.

Usage:
    cd autobio-review/AutoBio/autobio
    python ../autobio_isaaclab/scripts/record_mujoco_demos.py
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
from pathlib import Path

import mujoco
from mujoco.renderer import Renderer
import numpy as np
import imageio.v2 as imageio

# Add autobio to path
AUTOBIO_ROOT = Path(__file__).parent.parent.parent / "autobio"
sys.path.insert(0, str(AUTOBIO_ROOT))

from task import create_expert

OUTPUT_DIR = Path(__file__).parent.parent / "demos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("pickup", "pickup"),
    ("insert", "insert"),
    ("thermal_cycler_open", "thermal_cycler_open"),
    ("thermal_cycler_close", "thermal_cycler_close"),
    ("pipette", "pipette"),
    ("screw_loose", "screw_loose"),
    ("screw_tighten", "screw_tighten"),
    ("thermal_mixer", "thermal_mixer"),
    ("vortex_mixer", "vortex_mixer"),
    ("insert_centrifuge_5430", "insert_centrifuge_5430"),
]

WIDTH, HEIGHT = 1280, 720
FPS = 24


def record_task(task_name: str, short_name: str) -> dict:
    """Record one task's expert demo."""
    output_path = OUTPUT_DIR / f"{short_name}.mp4"
    status = {"task": short_name, "ok": False, "error": None}

    print(f"\n{'='*60}")
    print(f"Recording: {task_name}")
    print(f"{'='*60}")

    try:
        expert = create_expert(task_name)
        model = expert.model
        data = expert.data
        print(f"  model loaded, nq={model.nq}, ncam={model.ncam}")

        model.vis.global_.offheight = HEIGHT
        model.vis.global_.offwidth = WIDTH

        renderer = Renderer(model, height=HEIGHT, width=WIDTH)

        cameras = []
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                cameras.append(name)

        camera_name = cameras[0] if cameras else None
        print(f"  cameras: {cameras}, using: {camera_name}")

        # Capture frames by hooking into manager.step()
        frames = []
        frame_interval = max(1, int(1.0 / FPS / model.opt.timestep))
        step_counter = [0]

        original_step = expert.manager.step

        def hooked_step():
            original_step()
            step_counter[0] += 1
            if step_counter[0] % frame_interval == 0:
                if camera_name:
                    renderer.update_scene(data, camera=camera_name)
                else:
                    renderer.update_scene(data)
                frames.append(renderer.render().copy())

        expert.manager.step = hooked_step

        # Stub serializer for tasks that call self.serializer.finish()
        class StubSerializer:
            def finish(self): pass
            def record(self, info): pass
            def within_save_dir(self):
                class Ctx:
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                return Ctx()
        expert.serializer = StubSerializer()

        # Run expert
        expert.reset()
        expert.execute()

        renderer.close()

        if frames:
            imageio.mimwrite(str(output_path), frames, fps=FPS, codec="libx264",
                           output_params=["-crf", "18"])
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"  Wrote {len(frames)} frames -> {output_path.name} ({size_mb:.1f} MB)")
            status["ok"] = True
            status["frames"] = len(frames)
            status["size_mb"] = round(size_mb, 2)
        else:
            print(f"  WARNING: No frames captured")
            status["error"] = "no_frames"

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        status["error"] = str(e)[:300]

    return status


def main():
    print(f"AutoBio MuJoCo Video Recorder")
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Output: {OUTPUT_DIR}")

    results = []
    for task_name, short_name in TASKS:
        r = record_task(task_name, short_name)
        results.append(r)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print(f"  OK: {len(ok)}/{len(results)}")
    for r in ok:
        print(f"    {r['task']}: {r.get('frames', '?')} frames, {r.get('size_mb', '?')} MB")
    for r in fail:
        print(f"    FAIL {r['task']}: {r.get('error', '?')[:100]}")


if __name__ == "__main__":
    main()
