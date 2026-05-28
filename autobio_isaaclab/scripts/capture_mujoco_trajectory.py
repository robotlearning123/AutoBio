#!/usr/bin/env python3
"""Capture MuJoCo expert trajectory data (body positions + orientations).

Outputs per-frame numpy arrays for Isaac Sim hybrid rendering.

Usage:
    cd autobio-review/AutoBio/autobio
    source ../.venv-mj33/bin/activate
    MUJOCO_GL=egl python ../autobio_isaaclab/scripts/capture_mujoco_trajectory.py
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
import json
from pathlib import Path

import mujoco
import numpy as np

AUTOBIO_ROOT = Path(__file__).parent.parent.parent / "autobio"
sys.path.insert(0, str(AUTOBIO_ROOT))

from task import create_expert

OUTPUT_DIR = Path(__file__).parent.parent / "trajectories"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24

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


def capture_trajectory(task_name: str, short_name: str) -> dict:
    out_path = OUTPUT_DIR / f"{short_name}_traj.npz"
    status = {"task": short_name, "ok": False, "error": None}

    print(f"\n{'='*60}")
    print(f"Capturing: {task_name}")
    print(f"{'='*60}")

    try:
        expert = create_expert(task_name)
        model = expert.model
        data = expert.data

        body_names = []
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            body_names.append(name or f"_unnamed_{i}")

        frame_interval = max(1, int(1.0 / FPS / model.opt.timestep))
        step_counter = [0]

        all_xpos = []
        all_xmat = []
        all_qpos = []

        original_step = expert.manager.step

        def hooked_step():
            original_step()
            step_counter[0] += 1
            if step_counter[0] % frame_interval == 0:
                all_xpos.append(data.xpos.copy())
                all_xmat.append(data.xmat.reshape(-1, 3, 3).copy())
                all_qpos.append(data.qpos.copy())

        expert.manager.step = hooked_step

        class StubSerializer:
            def finish(self): pass
            def record(self, info): pass
            def within_save_dir(self):
                class Ctx:
                    def __enter__(self): return self
                    def __exit__(self, *a): pass
                return Ctx()
        expert.serializer = StubSerializer()

        expert.reset()
        expert.execute()

        if all_xpos:
            np.savez_compressed(str(out_path),
                                xpos=np.array(all_xpos),
                                xmat=np.array(all_xmat),
                                qpos=np.array(all_qpos),
                                body_names=np.array(body_names),
                                timestep=np.array([model.opt.timestep]),
                                frame_interval=np.array([frame_interval]))
            sz = out_path.stat().st_size / (1024 * 1024)
            print(f"  Captured {len(all_xpos)} frames -> {out_path.name} ({sz:.1f} MB)")
            print(f"  Bodies: {model.nbody}, Joints: {model.njnt}")
            status["ok"] = True
            status["frames"] = len(all_xpos)
            status["bodies"] = model.nbody
        else:
            status["error"] = "no_frames"

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        status["error"] = str(e)[:300]

    return status


def main():
    print("MuJoCo Trajectory Capture")
    print(f"  Tasks: {len(TASKS)}")
    print(f"  Output: {OUTPUT_DIR}")

    results = []
    for task_name, short_name in TASKS:
        r = capture_trajectory(task_name, short_name)
        results.append(r)

    print(f"\n{'='*60}")
    print("SUMMARY")
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    print(f"  OK: {len(ok)}/{len(results)}")
    for r in ok:
        print(f"    {r['task']}: {r.get('frames', '?')} frames, {r.get('bodies', '?')} bodies")
    for r in fail:
        print(f"    FAIL {r['task']}: {r.get('error', '?')[:100]}")


if __name__ == "__main__":
    main()
