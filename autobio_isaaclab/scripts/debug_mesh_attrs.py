import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from pathlib import Path
import numpy as np
BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_mesh_attrs.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.usd
    from pxr import Usd, UsdGeom, Sdf

    # Load both scenes and compare
    for scene, prim_path in [("pickup", "/World/pickup"), ("insert", "/World/insert")]:
        usd = str((BASE / f"usd_assets/{scene}/{scene}/{scene}_deinst.usda").resolve())
        add_reference_to_stage(usd_path=usd, prim_path=prim_path)

    world = World(physics_dt=1.0/120.0, rendering_dt=1.0/24.0)
    world.reset()
    stage = omni.usd.get_context().get_stage()

    for scene, prim_path in [("pickup", "/World/pickup"), ("insert", "/World/insert")]:
        lines.append(f"\n=== {scene} ===")
        root = stage.GetPrimAtPath(prim_path)
        lines.append(f"Root: {root.GetPath()} type={root.GetTypeName()} valid={root.IsValid()}")

        mesh_count = 0
        for p in Usd.PrimRange(root):
            if p.GetTypeName() == "Mesh" and mesh_count < 2:
                mesh_count += 1
                lines.append(f"\n  Mesh: {p.GetPath()}")

                mesh = UsdGeom.Mesh(p)

                # Points
                points = mesh.GetPointsAttr().Get()
                lines.append(f"    Points: {len(points) if points else 0}")

                # Face counts
                fc = mesh.GetFaceVertexCountsAttr().Get()
                lines.append(f"    Faces: {len(fc) if fc else 0}")

                # Extent
                ext = mesh.GetExtentAttr().Get()
                lines.append(f"    Extent: {ext}")

                # Display color
                dc = mesh.GetDisplayColorAttr()
                if dc:
                    colors = dc.Get()
                    lines.append(f"    DisplayColor: {colors}")

                # Material binding
                rel = p.GetRelationship("material:binding")
                if rel:
                    targets = rel.GetTargets()
                    lines.append(f"    Material binding: {targets}")
                    for t in targets:
                        mat_prim = stage.GetPrimAtPath(t)
                        lines.append(f"      Material valid: {mat_prim.IsValid()} type={mat_prim.GetTypeName()}")
                else:
                    lines.append(f"    No material binding")

                # Purpose
                purpose = UsdGeom.Imageable(p).GetPurposeAttr()
                if purpose:
                    lines.append(f"    Purpose: {purpose.Get()}")

                # Visibility
                vis = UsdGeom.Imageable(p).GetVisibilityAttr()
                if vis:
                    lines.append(f"    Visibility: {vis.Get()}")

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
