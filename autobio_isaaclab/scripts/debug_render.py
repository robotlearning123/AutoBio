#!/usr/bin/env python3
"""Debug rendering - add lights, check camera, dump frame stats."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from pathlib import Path
import numpy as np
import imageio.v2 as imageio

BASE = Path(__file__).parent.parent

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf

world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)

usd_path = str((BASE / "usd_assets/pickup/pickup/pickup.usda").resolve())
add_reference_to_stage(usd_path=usd_path, prim_path="/World/scene")
world.reset()

stage = omni.usd.get_context().get_stage()

# Add a distant light (like sunlight)
light = rep.create.light_distant(rotation=(30, 30, 0), intensity=3000)

# Also add a dome light for ambient
dome = rep.create.light_dome(intensity=500)

# Camera
cam = rep.create.camera(position=(2.0, 2.0, 1.8), look_at=(0, 0, 0.5))
rp = rep.create.render_product(cam, resolution=(1280, 720))
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach([rp])

# Warm up
for _ in range(10):
    world.step(render=True)

# Capture test frame
data = rgb_annot.get_data()
if data is not None and data.size > 0:
    frame = data[:, :, :3].copy()
    print(f"Frame shape: {frame.shape}")
    print(f"Mean: {frame.mean():.1f}, Std: {frame.std():.1f}")
    print(f"Min: {frame.min()}, Max: {frame.max()}")
    print(f"Non-zero pixels: {np.count_nonzero(frame)}/{frame.size}")
else:
    print("No frame data")

# Try writing a single frame
if data is not None:
    imageio.imwrite(str(BASE / "demos/test_isaac_frame.png"), frame)
    print("Wrote test frame to demos/test_isaac_frame.png")

app.close()
