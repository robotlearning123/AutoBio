#!/usr/bin/env python3
"""Create de-instanced USD files for Isaac Sim rendering.

Removes instanceable=true from prim definitions so meshes resolve properly.
"""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from pathlib import Path
BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_deinstance.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from pxr import Usd, UsdGeom, Sdf

    scenes = ["pickup", "insert", "mani_thermal_cycler", "mani_pipette",
              "lab_screw_all", "mani_thermal_mixer", "vortex_mixer",
              "insert_centrifuge_5430"]

    for scene in scenes:
        src = (BASE / f"usd_assets/{scene}/{scene}/{scene}.usda").resolve()
        if not src.exists():
            lines.append(f"SKIP: {scene} (no usda)")
            continue

        stage = Usd.Stage.Open(str(src))

        # Remove instanceable from all prims
        removed = 0
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                removed += 1

        # Export as de-instanced
        dst = BASE / f"usd_assets/{scene}/{scene}/{scene}_deinst.usda"
        stage.GetRootLayer().Export(str(dst))

        # Verify
        vstage = Usd.Stage.Open(str(dst))
        prim_count = sum(1 for _ in vstage.Traverse())
        mesh_count = sum(1 for p in vstage.Traverse() if p.GetTypeName() == "Mesh")
        lines.append(f"{scene}: removed {removed} instanceable, {prim_count} prims, {mesh_count} meshes")

        for p in vstage.Traverse():
            if p.GetTypeName() == "Mesh":
                lines.append(f"  MESH: {p.GetPath()}")
                if mesh_count > 5:
                    break

        stage = None
        vstage = None

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
