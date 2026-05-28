#!/usr/bin/env python3
"""Render Isaac Sim animated scene from MuJoCo trajectory — v5.

v5: camera matched to MuJoCo table_cam_front from XML scene files.
    focal_length 24mm (fovy=45 equiv). Debug logging for body mapping.

Usage:
    cd autobio-review/AutoBio/autobio_isaaclab
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
    /home/robot/miniconda3/envs/isaac5/bin/python scripts/record_isaac_v5.py [--task TASK]
"""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

import json
import sys
import re
import subprocess
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

BASE = Path(__file__).parent.parent
USD_ROOT = BASE / "usd_assets"
DEMO_DIR = BASE / "demos"
TRAJ_DIR = BASE / "trajectories"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1280, 720
FPS = 24
HUGE_THRESHOLD = 10.0

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

COLOR_MAP = {
    "black": (0.35, 0.35, 0.40),
    "vention_black": (0.05, 0.05, 0.1),
    "vention_white": (0.4, 0.4, 0.4),
    "vention_blue": (0.14, 0.30, 0.52),
    "centrifuge_10slot": (0.3, 0.4, 0.6),
    "centrifuge_body": (1.0, 1.0, 1.0),
    "centrifuge_cap": (0.93, 0.27, 0.06),
    "table": (0.35, 0.30, 0.25),
    "floor": (0.5, 0.6, 0.7),
    "default": (0.6, 0.6, 0.6),
}


def get_color_for_mesh(mesh_name: str, prim_path: str) -> tuple:
    n = mesh_name.lower()
    p = prim_path.lower()
    if "vx300s" in n or "d405" in n:
        return COLOR_MAP["black"]
    if "centrifuge_50ml_screw_cap" in n or ("cap" in n and "centrifuge" in p):
        return COLOR_MAP["centrifuge_cap"]
    if "centrifuge_50ml_screw_body" in n or "tube" in n:
        return COLOR_MAP["centrifuge_body"]
    if "centrifuge_10slot" in n or "pillar" in n or "plane" in n:
        return COLOR_MAP["centrifuge_10slot"]
    if "vention" in p or ("table" in n and "vx300s" not in n):
        return COLOR_MAP["table"]
    if "thermal_cycler" in n or "thermal_mixer" in n:
        return COLOR_MAP["vention_blue"]
    if "vortex" in n:
        return COLOR_MAP["vention_white"]
    if "pipette" in n or "tip" in n:
        return (0.9, 0.9, 0.95)
    if "screw" in n and "cap" not in n:
        return COLOR_MAP["centrifuge_cap"]
    return COLOR_MAP["default"]


def mujoco_name_to_usd_key(mj_name: str) -> str:
    """Convert MuJoCo body name to USD prim key for matching.

    Handles formats:
      '1/aloha:left/shoulder_link' -> 'left_shoulder_link'
      '/ur:shoulder_link'          -> 'shoulder_link'
      '/ur:dh:rh_base'            -> 'rh_base'
      '1/centrifuge_10slot'        -> 'centrifuge_10slot'
      '1/ur5e'                    -> 'ur5e'
    """
    s = mj_name.lstrip('/')
    # Split on colons to remove prefixes (e.g., 'ur:', 'aloha:left:')
    parts = s.split(':')
    # Take the last colon-delimited segment (the actual link name)
    last = parts[-1]
    # If it contains slashes, take the last slash-delimited segment
    last = last.split('/')[-1]
    # If there's a prefix before the colon, prepend it with underscore
    if len(parts) > 1:
        prefix = parts[-2].split('/')[-1]
        if prefix and prefix != last:
            return f"{prefix}_{last}"
    return last


def mat3_to_quat_wxyz(R):
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


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


def record_animated(task_name: str) -> dict:
    scene_name = TASKS[task_name]
    usd_path = find_deinst_usd(scene_name)
    output_path = DEMO_DIR / f"{task_name}_isaac_v5.mp4"
    result_path = DEMO_DIR / f"{task_name}_isaac_v5_result.json"
    traj_path = TRAJ_DIR / f"{task_name}_traj.npz"
    status = {"task": task_name, "scene": scene_name, "ok": False, "error": None}

    if not usd_path.exists():
        status["error"] = f"USD not found: {usd_path}"
        result_path.write_text(json.dumps(status))
        return status

    has_traj = traj_path.exists()
    if has_traj:
        traj = dict(np.load(str(traj_path), allow_pickle=True))
        xpos = traj["xpos"]
        xmat = traj["xmat"]
        body_names = [str(b) for b in traj["body_names"]]
        n_frames_total = xpos.shape[0]
        target_frames = min(120, n_frames_total // 10)
        target_frames = max(48, target_frames)
        sample_indices = np.linspace(0, n_frames_total - 1, target_frames, dtype=int)
    else:
        target_frames = 48
        sample_indices = None

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    try:
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.prims.xform_prim import XFormPrim
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Usd, UsdGeom, Vt, Gf, Sdf

        world = World(physics_dt=1.0 / 120.0, rendering_dt=1.0 / FPS)
        prim_root = f"/World/{task_name}"
        add_reference_to_stage(usd_path=str(usd_path.resolve()), prim_path=prim_root)
        world.reset()

        stage = omni.usd.get_context().get_stage()

        # 1. Hide huge meshes (Newton artifacts)
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

        # 2. Apply colors to meshes
        colored = 0
        for p in stage.Traverse():
            if p.GetTypeName() == "Mesh":
                mesh_name = p.GetName()
                prim_path_str = p.GetPath().pathString
                color = get_color_for_mesh(mesh_name, prim_path_str)
                color_attr = p.GetAttribute("primvars:displayColor")
                if not color_attr.IsValid():
                    color_attr = p.CreateAttribute(
                        "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
                color_attr.Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
                if color == COLOR_MAP["centrifuge_body"]:
                    op_attr = p.GetAttribute("primvars:displayOpacity")
                    if not op_attr.IsValid():
                        op_attr = p.CreateAttribute(
                            "primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray)
                    op_attr.Set(Vt.FloatArray([0.3]))
                colored += 1
        status["colored_meshes"] = colored

        # 3. Build name mapping: MuJoCo body -> USD XformPrim
        body_prim_map = {}
        if has_traj:
            root_prim = stage.GetPrimAtPath(prim_root)
            if root_prim.IsValid():
                xform_prims = {}
                for p in Usd.PrimRange(root_prim):
                    if p.GetTypeName() == "Xform":
                        xform_prims[p.GetName()] = p.GetPath().pathString

                # Pass 1: exact key match (with prefix)
                for i, bn in enumerate(body_names):
                    if bn == "world":
                        continue
                    usd_key = mujoco_name_to_usd_key(bn)
                    if usd_key in xform_prims:
                        body_prim_map[i] = xform_prims[usd_key]

                # Pass 2: try without prefix (e.g., 'shoulder_link' instead of 'ur_shoulder_link')
                for i, bn in enumerate(body_names):
                    if i in body_prim_map or bn == "world":
                        continue
                    s = bn.lstrip('/')
                    parts = s.split(':')
                    last_part = parts[-1].split('/')[-1]
                    if last_part in xform_prims:
                        body_prim_map[i] = xform_prims[last_part]

                # Pass 3: fuzzy match on last segment
                for i, bn in enumerate(body_names):
                    if i in body_prim_map or bn == "world":
                        continue
                    short = bn.split("/")[-1].split(":")[-1]
                    for xname, xpath in xform_prims.items():
                        if short in xname or xname.endswith(short):
                            body_prim_map[i] = xpath
                            break
        status["mapped_bodies"] = len(body_prim_map)

        # Debug: log first 10 body mappings
        mapping_debug = []
        for i, bn in enumerate(body_names):
            if i in body_prim_map:
                mapping_debug.append(f"  body[{i}] {bn} -> {body_prim_map[i]}")
        status["body_map_debug"] = mapping_debug[:20]
        for line in mapping_debug[:10]:
            print(line)

        # 4. Setup rendering — v5: cameras from MuJoCo XML table_cam_front
        # MuJoCo camera pos + quat(0.819,0.574,0,0) → forward dir [0, 0.94, -0.34]
        # look_at = ray intersection with z=0.8 (table height)
        cam_overrides = {
            "pickup": {"pos": (0, -1.2, 1.45), "target": (0.0, 0.59, 0.8), "focal_length": 24},
            "insert": {"pos": (0, -1.2, 1.45), "target": (0.0, 0.59, 0.8), "focal_length": 24},
            "pipette": {"pos": (0, -1.2, 1.35), "target": (0.0, 0.32, 0.8), "focal_length": 24},
            "thermal_cycler_open": {"pos": (0, -1.5, 1.65), "target": (0.0, 0.84, 0.8), "focal_length": 24},
            "thermal_cycler_close": {"pos": (0, -1.5, 1.65), "target": (0.0, 0.84, 0.8), "focal_length": 24},
            "screw_loose": {"pos": (0, -1.5, 1.65), "target": (0.0, 0.84, 0.8), "focal_length": 24},
            "screw_tighten": {"pos": (0, -1.5, 1.65), "target": (0.0, 0.84, 0.8), "focal_length": 24},
            "thermal_mixer": {"pos": (0, -1.5, 1.65), "target": (0.0, 0.84, 0.8), "focal_length": 24},
            "vortex_mixer": {"pos": (0, -1.2, 1.45), "target": (0.0, 0.59, 0.8), "focal_length": 24},
            "insert_centrifuge_5430": {"pos": (0, -1.5, 1.65), "target": (0.0, 0.84, 0.8), "focal_length": 24},
        }
        cam_cfg = cam_overrides.get(task_name, {})
        light_mult = cam_cfg.get("light_mult", 2.0)
        rep.create.light(light_type="Distant", intensity=int(800 * light_mult), rotation=(45, 45, 0))
        rep.create.light(light_type="Distant", intensity=int(400 * light_mult), rotation=(-30, -60, 0))
        rep.create.light(light_type="Dome", intensity=int(120 * light_mult))
        cam_pos = cam_cfg.get("pos", (1.5, 1.5, 1.5))
        cam_target = cam_cfg.get("target", (0, 0, 0.8))
        cam_focal = cam_cfg.get("focal_length", 35)

        cam = rep.create.camera(position=cam_pos, look_at=cam_target, focal_length=cam_focal)
        rp = rep.create.render_product(cam, resolution=(WIDTH, HEIGHT))
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([rp])

        for _ in range(20):
            world.step(render=True)

        # 5. Animated rendering using direct USD transforms
        #    (bypass physics via app.update() to avoid kinematic override)
        frames = []

        def set_world_transform_matrix(prim, world_mat):
            xformable = UsdGeom.Xformable(prim)
            parent_prim = prim.GetParent()
            if parent_prim and parent_prim.IsValid():
                cache = UsdGeom.XformCache()
                parent_world = cache.GetLocalToWorldTransform(parent_prim)
                parent_inv = parent_world.GetInverse()
                local_mat = parent_inv * world_mat
            else:
                local_mat = world_mat
            xformable.ClearXformOpOrder()
            op = xformable.AddTransformOp()
            op.Set(local_mat)

        if has_traj and body_prim_map:
            for fi, t_idx in enumerate(sample_indices):
                for body_idx, prim_path_str in body_prim_map.items():
                    if t_idx >= xpos.shape[0]:
                        continue
                    prim = stage.GetPrimAtPath(prim_path_str)
                    if not prim.IsValid():
                        continue
                    pos = xpos[t_idx, body_idx]
                    R = xmat[t_idx, body_idx]
                    mat = Gf.Matrix4d(
                        R[0, 0], R[0, 1], R[0, 2], 0.0,
                        R[1, 0], R[1, 1], R[1, 2], 0.0,
                        R[2, 0], R[2, 1], R[2, 2], 0.0,
                        pos[0], pos[1], pos[2], 1.0,
                    )
                    set_world_transform_matrix(prim, mat)

                # Debug: log first frame transforms for first 5 mapped bodies
                if fi == 0:
                    t_idx_0 = sample_indices[0]
                    for body_idx in list(body_prim_map.keys())[:5]:
                        p = xpos[t_idx_0, body_idx]
                        print(f"  frame0 body[{body_idx}] pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

                app.update()
                data = rgb_annot.get_data()
                if data is not None and data.size > 4:
                    frames.append(data[:, :, :3].copy())
        else:
            for _ in range(target_frames):
                app.update()
                data = rgb_annot.get_data()
                if data is not None and data.size > 4:
                    frames.append(data[:, :, :3].copy())

        if frames:
            first_mean = float(np.mean(frames[0])) if frames else 0.0
            if len(frames) > 1:
                diff = float(np.mean(np.abs(
                    frames[-1].astype(float) - frames[0].astype(float))))
            else:
                diff = 0.0

            imageio.mimwrite(str(output_path), frames, fps=FPS, codec="libx264",
                           output_params=["-crf", "18"])
            size_mb = output_path.stat().st_size / (1024 * 1024)
            status["ok"] = True
            status["frames"] = len(frames)
            status["size_mb"] = round(size_mb, 2)
            status["first_mean"] = round(first_mean, 1)
            status["frame_diff"] = round(diff, 1)
            status["animated"] = has_traj and len(body_prim_map) > 0
        else:
            status["error"] = "no_frames"

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

    python = sys.executable
    script = str(Path(__file__).resolve())

    if args.task:
        if args.task not in TASKS:
            print(f"Unknown task: {args.task}")
            sys.exit(1)
        r = record_animated(args.task)
        print(f"{r['task']}: ok={r['ok']} mean={r.get('first_mean','?')} "
              f"size={r.get('size_mb','?')}MB hidden={r.get('hidden_meshes',0)} "
              f"colored={r.get('colored_meshes',0)} mapped={r.get('mapped_bodies',0)} "
              f"diff={r.get('frame_diff',0)} anim={r.get('animated',False)}")
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
                timeout=300,
            )
            if result_file.exists():
                r = json.loads(result_file.read_text())
            else:
                r = {"task": task_name, "ok": False, "error": "no result file"}
            results.append(r)
            print(f"  {r['task']}: ok={r['ok']} mean={r.get('first_mean','?')} "
                  f"size={r.get('size_mb','?')}MB hidden={r.get('hidden_meshes',0)} "
                  f"colored={r.get('colored_meshes',0)} mapped={r.get('mapped_bodies',0)} "
                  f"diff={r.get('frame_diff',0)} anim={r.get('animated',False)}")

        ok = [r for r in results if r["ok"]]
        fail = [r for r in results if not r["ok"]]
        summary_path = DEMO_DIR / "isaac_v5_summary.json"
        summary_path.write_text(json.dumps(results, indent=2))
        print(f"\nSUMMARY: {len(ok)}/{len(results)} OK")
        for r in ok:
            print(f"  {r['task']}: {r.get('frames','?')} frames, "
                  f"{r.get('size_mb','?')} MB, mean={r.get('first_mean','?')}, "
                  f"diff={r.get('frame_diff','?')}, anim={r.get('animated',False)}")
        for r in fail:
            print(f"  FAIL {r['task']}: {r.get('error','?')[:100]}")


if __name__ == "__main__":
    main()
