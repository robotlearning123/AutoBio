#!/usr/bin/env python3
"""Flatten Newton USD to resolve all references and instances."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from pathlib import Path
BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_flatten.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from pxr import Usd, UsdGeom
    import omni.usd

    for scene in ["pickup"]:
        src = str((BASE / f"usd_assets/{scene}/{scene}/{scene}.usda").resolve())
        lines.append(f"Source: {src}")

        # Open as standalone stage, then flatten
        stage = Usd.Stage.Open(src)
        prim_count_before = sum(1 for _ in stage.Traverse())
        mesh_count_before = sum(1 for p in stage.Traverse() if p.GetTypeName() == "Mesh")
        lines.append(f"  Before flatten: {prim_count_before} prims, {mesh_count_before} meshes")

        # Flatten to resolve all references
        flat_path = str((BASE / f"usd_assets/{scene}/{scene}/{scene}_flat.usda").resolve())
        stage.Export(flat_path)
        lines.append(f"  Exported to: {flat_path}")

        # Re-open the flattened version
        flat_stage = Usd.Stage.Open(flat_path)
        prim_count_after = sum(1 for _ in flat_stage.Traverse())
        mesh_count_after = sum(1 for p in flat_stage.Traverse() if p.GetTypeName() == "Mesh")
        lines.append(f"  After flatten: {prim_count_after} prims, {mesh_count_after} meshes")

        # Show mesh prims
        for p in flat_stage.Traverse():
            if p.GetTypeName() == "Mesh":
                lines.append(f"    MESH: {p.GetPath()}")

        flat_stage = None
        stage = None

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
