#!/usr/bin/env python3
"""Render Isaac Sim videos using Newton USD meshes + MuJoCo kinematic trajectories.

Loads Newton-format USD assets (which have visual meshes) and applies MuJoCo
body transforms kinematically for each frame. No physics simulation needed.

Usage (one task per process due to SimulationApp singleton):
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    conda run -n isaac5 --cwd autobio_isaaclab \
    python scripts/record_isaac_hybrid.py --task pickup
"""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

import json
import re
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

USD_ROOT = Path(__file__).parent.parent / "usd_assets"
TRAJ_DIR = Path(__file__).parent.parent / "trajectories"
DEMO_DIR = Path(__file__).parent.parent / "demos"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
FPS = 24

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


def mujoco_body_name_to_usd_prim_name(mj_name: str) -> str:
    """Map MuJoCo body name to USD prim name (leaf segment)."""
    if not mj_name or mj_name.startswith("_unnamed"):
        return None
    # e.g. "1/aloha:left/base_link" -> "left_base_link"
    # e.g. "2/centrifuge_50ml_screw_body" -> "centrifuge_50ml_screw_body"
    # e.g. "aloha1" -> "aloha1"
    # e.g. "centrifuge_50ml_screw_cap_wrap" -> "centrifuge_50ml_screw_cap_wrap"
    if "/" in mj_name:
        return mj_name.split("/")[-1].replace(":", "_")
    if ":" in mj_name:
        return mj_name.replace(":", "_")
    return mj_name


def find_prim_by_name(stage, root_path: str, name: str):
    """Find a USD prim by name under a root path."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in root.GetAllChildren():
        if prim.GetName() == name:
            return prim
        result = _find_prim_recursive(prim, name)
        if result:
            return result
    return None


def _find_prim_recursive(prim, name: str):
    if prim.GetName() == name:
        return prim
    for child in prim.GetAllChildren():
        result = _find_prim_recursive(child, name)
        if result:
            return result
    return None


def build_body_prim_map(stage, root_prim_path: str, body_names: list) -> dict:
    """Map MuJoCo body indices to USD prim paths."""
    mapping = {}
    for i, mj_name in enumerate(body_names):
        usd_name = mujoco_body_name_to_usd_prim_name(mj_name)
        if usd_name is None:
            continue
        prim = find_prim_by_name(stage, root_prim_path, usd_name)
        if prim and prim.IsValid():
            mapping[i] = prim.GetPath()
    return mapping


def set_world_transform(prim, translation, rotation_matrix):
    """Set a prim's transform in world space."""
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation.tolist()))

    mat = rotation_matrix.flatten()[:9].tolist()
    rot = Gf.Matrix3d(
        mat[0], mat[1], mat[2],
        mat[3], mat[4], mat[5],
        mat[6], mat[7], mat[8]
    )
    orient_op = xformable.AddOrientOp()
    quat = rot.GetQuaternion()
    orient_op.Set(Gf.Quatf(quat.GetReal(), *quat.GetImaginary()))


def find_usd_path(scene_name: str) -> Path:
    """Find USD file — prefer .usda (text) over .usd (binary crate)."""
    scene_dir = USD_ROOT / scene_name / scene_name
    usda = scene_dir / f"{scene_name}.usda"
    usd = scene_dir / f"{scene_name}.usd"
    if usda.exists():
        return usda
    if usd.exists():
        return usd
    return usd


def record_task(task_name: str) -> dict:
    scene_name = TASKS[task_name]
    usd_path = find_usd_path(scene_name)
    traj_path = TRAJ_DIR / f"{task_name}_traj.npz"
    output_path = DEMO_DIR / f"{task_name}_isaac.mp4"
    result_path = DEMO_DIR / f"{task_name}_isaac_result.json"
    status = {"task": task_name, "ok": False, "error": None}

    if not usd_path.exists():
        status["error"] = f"USD not found: {usd_path}"
        result_path.write_text(json.dumps(status))
        return status

    if not traj_path.exists():
        status["error"] = f"Trajectory not found: {traj_path}"
        result_path.write_text(json.dumps(status))
        return status

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    try:
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom

        traj = np.load(str(traj_path))
        xpos = traj["xpos"]
        xmat = traj["xmat"]
        body_names = list(traj["body_names"])
        nframes = xpos.shape[0]
        print(f"  Trajectory: {nframes} frames, {len(body_names)} bodies")

        world = World(physics_dt=1.0/120.0, rendering_dt=1.0/FPS)

        prim_path = f"/World/{task_name}"
        add_reference_to_stage(usd_path=str(usd_path), prim_path=prim_path)
        world.reset()

        stage = omni.usd.get_context().get_stage()

        body_prim_map = build_body_prim_map(stage, prim_path, body_names)
        print(f"  Mapped {len(body_prim_map)}/{len(body_names)} bodies to USD prims")
        for bi, pp in sorted(body_prim_map.items())[:5]:
            print(f"    body[{bi}] {body_names[bi]} -> {pp}")
        if len(body_prim_map) > 5:
            print(f"    ... +{len(body_prim_map)-5} more")

        cam = rep.create.camera(position=(2.0, 2.0, 1.8), look_at=(0, 0, 0.5))
        rp = rep.create.render_product(cam, resolution=(WIDTH, HEIGHT))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([rp])

        for _ in range(5):
            world.step(render=True)

        frames = []
        for frame_i in range(nframes):
            for body_i, prim_path_str in body_prim_map.items():
                if body_i < xpos.shape[1]:
                    prim = stage.GetPrimAtPath(prim_path_str)
                    if prim.IsValid():
                        set_world_transform(prim, xpos[frame_i, body_i], xmat[frame_i, body_i])

            world.step(render=True)
            data = rgb_annot.get_data()
            if data is not None and data.size > 0:
                frame = data[:, :, :3].copy()
                frames.append(frame)

            if frame_i % 50 == 0:
                print(f"    Frame {frame_i}/{nframes}")

        if frames:
            imageio.mimwrite(str(output_path), frames, fps=FPS, codec="libx264",
                           output_params=["-crf", "18"])
            size_mb = output_path.stat().st_size / (1024 * 1024)
            status["ok"] = True
            status["frames"] = len(frames)
            status["size_mb"] = round(size_mb, 2)
            print(f"  Wrote {len(frames)} frames -> {output_path.name} ({size_mb:.1f} MB)")
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
        print(f"Available: {list(TASKS.keys())}")
        sys.exit(1)
    result = record_task(args.task)
    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
