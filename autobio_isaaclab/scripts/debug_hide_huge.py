import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_hide_huge.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom, Gf

    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)
    usd = str((BASE / "usd_assets/insert/insert/insert_deinst.usda").resolve())
    add_reference_to_stage(usd_path=usd, prim_path="/World/insert")
    world.reset()
    stage = omni.usd.get_context().get_stage()

    # Hide all Mesh prims with world bounds > 10m in any dimension
    hidden = 0
    for p in stage.Traverse():
        if p.GetTypeName() == "Mesh":
            bbox = UsdGeom.Imageable(p).ComputeWorldBound(Usd.TimeCode.Default(), "default", "default")
            if bbox:
                rng = bbox.GetRange()
                sx = rng.GetMax()[0] - rng.GetMin()[0]
                sy = rng.GetMax()[1] - rng.GetMin()[1]
                sz = rng.GetMax()[2] - rng.GetMin()[2]
                if sx > 10 or sy > 10 or sz > 10:
                    UsdGeom.Imageable(p).GetVisibilityAttr().Set("invisible")
                    hidden += 1
                    lines.append(f"  Hidden: {p.GetPath().pathString} ({sx:.0f}x{sy:.0f}x{sz:.0f})")

    lines.append(f"Hidden {hidden} huge meshes")

    # Check new bounds
    root = stage.GetPrimAtPath("/World/insert")
    bbox = UsdGeom.Imageable(root).ComputeWorldBound(Usd.TimeCode.Default(), "default", "default")
    if bbox:
        rng = bbox.GetRange()
        lines.append(f"New bounds: ({rng.GetMin()[0]:.2f},{rng.GetMin()[1]:.2f},{rng.GetMin()[2]:.2f}) to ({rng.GetMax()[0]:.2f},{rng.GetMax()[1]:.2f},{rng.GetMax()[2]:.2f})")

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
        r_mean = float(np.mean(data[:, :, 0]))
        g_mean = float(np.mean(data[:, :, 1]))
        b_mean = float(np.mean(data[:, :, 2]))
        lines.append(f"After hiding huge: R={r_mean:.1f} G={g_mean:.1f} B={b_mean:.1f}")
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
