import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import sys

BASE = Path(__file__).parent.parent
DEMO_DIR = BASE / "demos"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

task = sys.argv[1] if len(sys.argv) > 1 else "insert"

SCENES = {
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

scene = SCENES[task]
usd_file = BASE / f"usd_assets/{scene}/{scene}/{scene}_deinst.usda"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom

world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)
add_reference_to_stage(usd_path=str(usd_file.resolve()), prim_path=f"/World/{task}")
world.reset()

rep.create.light(light_type="Distant", intensity=5000, rotation=(45, 45, 0))
rep.create.light(light_type="Dome", intensity=1000)
cam = rep.create.camera(position=(3.0, 3.0, 2.0), look_at=(0, 0, 0.5))
rp = rep.create.render_product(cam, resolution=(640, 480))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

for _ in range(20):
    world.step(render=True)

data = rgb.get_data()
print(f"data shape={data.shape if data is not None else None}")
if data is not None and data.size > 4:
    mean = float(np.mean(data[:, :, :3]))
    print(f"first frame mean={mean:.1f}")
else:
    print("no valid data")

frames = []
for i in range(48):
    world.step(render=True)
    data = rgb.get_data()
    if data is not None and data.size > 4:
        frames.append(data[:, :, :3].copy())

if frames:
    out_path = DEMO_DIR / f"{task}_isaac.mp4"
    imageio.mimwrite(str(out_path), frames, fps=24, codec="libx264", output_params=["-crf", "18"])
    print(f"saved {len(frames)} frames to {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
else:
    print("no frames captured")

app.close()
