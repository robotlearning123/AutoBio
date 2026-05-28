import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_mat_strip.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom, Sdf, Vt, Gf

    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)
    add_reference_to_stage(
        usd_path=str((BASE / "usd_assets/insert/insert/insert_deinst.usda").resolve()),
        prim_path="/World/insert",
    )
    world.reset()
    stage = omni.usd.get_context().get_stage()

    # Strip all material bindings on all prims
    stripped = 0
    for p in stage.Traverse():
        for rel_name in ["material:binding", "material:binding:physics"]:
            rel = p.GetRelationship(rel_name)
            if rel and rel.GetTargets():
                rel.ClearTargets(True)
                stripped += 1

    lines.append(f"Stripped {stripped} material bindings")

    rep.create.light(light_type="Distant", intensity=5000, rotation=(45, 45, 0))
    rep.create.light(light_type="Dome", intensity=1000)
    cam = rep.create.camera(position=(3.0, 3.0, 2.0), look_at=(0, 0, 0.5))
    rp = rep.create.render_product(cam, resolution=(640, 480))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

    for _ in range(20):
        world.step(render=True)

    data = rgb.get_data()
    if data is not None and data.size > 4:
        rgb_mean = float(np.mean(data[:, :, :3]))
        a_mean = float(np.mean(data[:, :, 3]))
        lines.append(f"After stripping materials: RGB mean={rgb_mean:.1f} A mean={a_mean:.1f}")
    else:
        lines.append("No data")

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
