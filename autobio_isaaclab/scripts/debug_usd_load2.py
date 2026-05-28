#!/usr/bin/env python3
"""Debug USD loading - try USDA vs USDC, absolute paths."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from pathlib import Path
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import omni.usd
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core import World
from pxr import Usd, UsdGeom

world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)

BASE = Path(__file__).parent.parent

# Test 1: USDA with absolute path
usd_path = str(BASE / "usd_assets/pickup/pickup/pickup.usda")
add_reference_to_stage(usd_path=usd_path, prim_path="/World/test_usda")
world.reset()

stage = omni.usd.get_context().get_stage()

lines = []
for test_name, prim_path in [("USDA", "/World/test_usda")]:
    root = stage.GetPrimAtPath(prim_path)
    prim_count = 0
    geom_count = 0
    mesh_paths = []
    for prim in Usd.PrimRange(root):
        prim_count += 1
        pt = prim.GetTypeName()
        if pt in ("Mesh", "Plane", "Capsule", "BasisCurves", "Cube", "Sphere", "Cone", "Cylinder"):
            geom_count += 1
            if len(mesh_paths) < 10:
                mesh_paths.append(f"  GEOM: {prim.GetPath()} type={pt}")
    lines.append(f"\n{test_name}: path={prim_path}")
    lines.append(f"  Root valid: {root.IsValid()}")
    lines.append(f"  Total prims: {prim_count}")
    lines.append(f"  Geom prims: {geom_count}")
    children = list(root.GetAllChildren()) if root.IsValid() else []
    lines.append(f"  Root children: {len(children)}")
    for c in children[:5]:
        lines.append(f"    {c.GetPath()} ({c.GetTypeName()})")
    for mp in mesh_paths:
        lines.append(mp)

with open(str(BASE / "demos/debug_usd_load2.txt"), "w") as f:
    f.write("\n".join(lines))

app.close()
