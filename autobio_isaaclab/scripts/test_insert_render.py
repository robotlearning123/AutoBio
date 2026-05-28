import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from pathlib import Path
import numpy as np
import imageio.v2 as imageio

BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_insert_render.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom

    # Load insert scene
    usd = str((BASE / "usd_assets/insert/insert/insert_deinst.usda").resolve())
    add_reference_to_stage(usd_path=usd, prim_path="/World/insert")
    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)
    world.reset()

    # Count meshes on Isaac Sim stage
    stage = omni.usd.get_context().get_stage()
    mesh_count = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    prim_count = sum(1 for _ in stage.Traverse())
    lines.append(f"Stage: {prim_count} prims, {mesh_count} meshes")

    # List some meshes
    for p in stage.Traverse():
        if p.GetTypeName() == "Mesh":
            lines.append(f"  MESH: {p.GetPath()}")
            vis = UsdGeom.Imageable(p).ComputeVisibility()
            purpose = UsdGeom.Imageable(p).ComputePurpose()
            lines.append(f"    vis={vis} purpose={purpose}")
            if mesh_count > 5:
                break

    # Add lights
    rep.create.light(light_type="Distant", intensity=5000, rotation=(45, 45, 0))
    rep.create.light(light_type="Dome", intensity=1000)

    # Try multiple camera positions
    for cam_pos, cam_look, label in [
        ((2.0, 2.0, 1.8), (0, 0, 0.5), "default"),
        ((1.5, 0.0, 1.5), (0, 0, 0.8), "front_close"),
        ((0.5, 0.5, 2.0), (0, 0, 0), "above"),
        ((5.0, 5.0, 5.0), (0, 0, 0), "far"),
    ]:
        cam = rep.create.camera(position=cam_pos, look_at=cam_look)
        rp = rep.create.render_product(cam, resolution=(1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])

        for _ in range(15):
            world.step(render=True)

        data = rgb.get_data()
        if data is not None:
            mean = float(np.mean(data[:, :, :3]))
            lines.append(f"Camera {label} pos={cam_pos}: mean={mean:.1f}")
        else:
            lines.append(f"Camera {label}: no data")

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
