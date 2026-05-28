import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_sphere.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom, Vt, Gf

    usd = str((BASE / "usd_assets/insert/insert/insert_deinst.usda").resolve())
    add_reference_to_stage(usd_path=usd, prim_path="/World/insert")
    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)
    world.reset()

    # Add a bright red sphere at origin
    rep.create.sphere(position=(0, 0, 0.5), semantics=[("class", "test_sphere")])

    rep.create.light(light_type="Distant", intensity=5000, rotation=(45, 45, 0))
    rep.create.light(light_type="Dome", intensity=1000)

    cam = rep.create.camera(position=(3.0, 3.0, 2.0), look_at=(0, 0, 0.5))
    rp = rep.create.render_product(cam, resolution=(640, 480))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])

    for _ in range(15):
        world.step(render=True)

    data = rgb.get_data()
    if data is not None:
        mean_val = float(np.mean(data[:, :, :3]))
        max_val = float(np.max(data[:, :, :3]))
        lines.append(f"With red sphere: mean={mean_val:.1f} max={max_val:.1f}")
    else:
        lines.append("No frame data")

except Exception as e:
    import traceback
    lines.append(f"ERROR: {e}")
    lines.extend(traceback.format_exc().split("\n"))
finally:
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(lines))
    except Exception as e2:
        # Last resort - print
        print(f"WRITE FAILED: {e2}")
        print("\n".join(lines))
    app.close()
