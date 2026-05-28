import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from pathlib import Path
BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_bounds.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from pxr import Usd, UsdGeom, Gf

    for scene in ["pickup", "insert", "mani_thermal_cycler", "mani_pipette"]:
        usd = str((BASE / f"usd_assets/{scene}/{scene}/{scene}_deinst.usda").resolve())
        stage = Usd.Stage.Open(usd)
        lines.append(f"\n{scene}:")

        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                mesh = UsdGeom.Mesh(prim)
                extent_attr = mesh.GetExtentAttr()
                if extent_attr:
                    extent = extent_attr.Get()
                    if extent and len(extent) == 2:
                        lines.append(f"  {prim.GetPath()}: extent={extent[0]} to {extent[1]}")
                points_attr = mesh.GetPointsAttr()
                if points_attr:
                    points = points_attr.Get()
                    if points:
                        lines.append(f"    {len(points)} points")
                break  # just first mesh

        # Also check root transform
        root = stage.GetDefaultPrim()
        xform = UsdGeom.Xformable(root)
        if xform:
            mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = mat.ExtractTranslation()
            lines.append(f"  Root world pos: {t}")

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
