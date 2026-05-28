#!/usr/bin/env python3
"""Debug USD loading in Isaac Sim."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from pxr import Usd, UsdGeom

world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)

usd_path = "usd_assets/pickup/pickup/pickup.usd"
add_reference_to_stage(usd_path=usd_path, prim_path="/World/test")
world.reset()

stage = omni.usd.get_context().get_stage()

lines = []
prim_count = 0
geom_count = 0
for prim in stage.Traverse():
    prim_count += 1
    pt = prim.GetTypeName()
    if pt in ("Mesh", "Plane", "Capsule", "BasisCurves", "Points", "Cube", "Sphere", "Cone", "Cylinder"):
        geom_count += 1
        if geom_count <= 20:
            lines.append(f"  GEOM: {prim.GetPath()} type={pt}")

lines.append(f"Total prims: {prim_count}")
lines.append(f"Geom prims: {geom_count}")

root = stage.GetPrimAtPath("/World/test")
lines.append(f"Root valid: {root.IsValid()}")
if root.IsValid():
    children = list(root.GetAllChildren())
    lines.append(f"Root children: {len(children)}")
    for c in children[:10]:
        lines.append(f"  {c.GetPath()} ({c.GetTypeName()})")

# Check payload resolution
geom_prim = stage.GetPrimAtPath("/World/test/Geometry")
lines.append(f"Geometry prim valid: {geom_prim.IsValid()}")
if geom_prim.IsValid():
    for gc in list(geom_prim.GetAllChildren())[:5]:
        lines.append(f"  {gc.GetPath()} ({gc.GetTypeName()})")

with open("demos/debug_usd_load.txt", "w") as f:
    f.write("\n".join(lines))

app.close()
