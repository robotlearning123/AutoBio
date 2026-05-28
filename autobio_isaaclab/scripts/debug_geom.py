#!/usr/bin/env python3
"""Debug geometries.usd structure."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from pathlib import Path
BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_geom.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from pxr import Usd, UsdGeom
    import omni.usd

    # Open geometries.usd directly as a new stage
    geom_path = str((BASE / "usd_assets/pickup/pickup/payloads/geometries.usd").resolve())
    stage = Usd.Stage.Open(geom_path)
    lines.append(f"Opened: {geom_path}")

    prim_count = 0
    mesh_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        pt = prim.GetTypeName()
        if pt == "Mesh":
            mesh_count += 1
            if mesh_count <= 10:
                lines.append(f"  MESH: {prim.GetPath()}")
        if prim_count <= 30:
            lines.append(f"  {prim.GetPath()} ({pt})")

    lines.append(f"Total prims: {prim_count}, Meshes: {mesh_count}")

    # Also check instances.usda
    inst_path = str((BASE / "usd_assets/pickup/pickup/payloads/instances.usda").resolve())
    stage2 = Usd.Stage.Open(inst_path)
    lines.append(f"\nOpened: {inst_path}")
    for prim in stage2.Traverse():
        lines.append(f"  {prim.GetPath()} ({prim.GetTypeName()})")

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
