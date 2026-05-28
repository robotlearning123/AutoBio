#!/usr/bin/env python3
"""Debug rendering - write all output to file."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from pathlib import Path
import numpy as np
import imageio.v2 as imageio

BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_render.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom

    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)

    usd_path = str((BASE / "usd_assets/pickup/pickup/pickup.usda").resolve())
    lines.append(f"USD path: {usd_path}")

    add_reference_to_stage(usd_path=usd_path, prim_path="/World/scene")
    world.reset()

    stage = omni.usd.get_context().get_stage()
    prim_count = sum(1 for _ in stage.Traverse())
    geom_count = sum(1 for p in stage.Traverse() if p.GetTypeName() in ("Mesh", "Plane", "Capsule", "Sphere", "Cube", "Cylinder"))
    lines.append(f"Prims: {prim_count}, Geoms: {geom_count}")

    # Check for instanced mesh prims
    mesh_prims = [p for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    lines.append(f"Mesh prims: {len(mesh_prims)}")
    for mp in mesh_prims[:5]:
        vis = UsdGeom.Imageable(mp).ComputeVisibility()
        purpose = UsdGeom.Imageable(mp).ComputePurpose()
        lines.append(f"  {mp.GetPath()} vis={vis} purpose={purpose}")

    # Add lights
    light = rep.create.light_distant(rotation=(30, 30, 0), intensity=3000)
    dome = rep.create.light_dome(intensity=500)
    lines.append("Added distant + dome lights")

    # Camera
    cam = rep.create.camera(position=(2.0, 2.0, 1.8), look_at=(0, 0, 0.5))
    rp = rep.create.render_product(cam, resolution=(1280, 720))
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annot.attach([rp])

    # Warm up
    for i in range(15):
        world.step(render=True)
    lines.append("Warmed up 15 steps")

    # Capture
    data = rgb_annot.get_data()
    if data is not None:
        lines.append(f"Data shape: {data.shape}, dtype: {data.dtype}")
        frame = data[:, :, :3].copy()
        lines.append(f"Frame mean: {frame.mean():.1f}, std: {frame.std():.1f}, min: {frame.min()}, max: {frame.max()}")
        lines.append(f"Non-zero: {np.count_nonzero(frame)}/{frame.size}")
        imageio.imwrite(str(BASE / "demos/test_frame.png"), frame)
        lines.append("Wrote test_frame.png")
    else:
        lines.append("data is None!")

except Exception as e:
    import traceback
    lines.append(f"ERROR: {e}")
    lines.extend(traceback.format_exc().split("\n"))
finally:
    try:
        OUT.write_text("\n".join(lines))
    except:
        pass
    app.close()
