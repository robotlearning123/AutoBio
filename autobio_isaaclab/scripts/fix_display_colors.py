#!/usr/bin/env python3
"""Fix display colors in de-instanced USD files — set black meshes to gray."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

from pathlib import Path
import numpy as np
BASE = Path(__file__).parent.parent
OUT = BASE / "demos" / "debug_fix_colors.txt"

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

lines = []
try:
    from pxr import Usd, UsdGeom, Gf, Sdf, Vt

    scenes = ["pickup", "insert", "mani_thermal_cycler", "mani_pipette",
              "lab_screw_all", "mani_thermal_mixer", "vortex_mixer",
              "insert_centrifuge_5430"]

    DEFAULT_COLOR = Gf.Vec3f(0.5, 0.5, 0.5)

    for scene in scenes:
        src = BASE / f"usd_assets/{scene}/{scene}/{scene}_deinst.usda"
        if not src.exists():
            lines.append(f"SKIP: {scene}")
            continue

        stage = Usd.Stage.Open(str(src))
        fixed = 0
        total = 0

        for prim in stage.Traverse():
            if prim.GetTypeName() == "Mesh":
                total += 1
                mesh = UsdGeom.Mesh(prim)
                dc_attr = mesh.GetDisplayColorAttr()
                if dc_attr:
                    colors = dc_attr.Get()
                    if colors and len(colors) > 0:
                        c = colors[0]
                        if c[0] == 0 and c[1] == 0 and c[2] == 0:
                            dc_attr.Set(Vt.Vec3fArray([DEFAULT_COLOR]))
                            fixed += 1

        dst = BASE / f"usd_assets/{scene}/{scene}/{scene}_deinst.usda"
        stage.GetRootLayer().Export(str(dst))
        lines.append(f"{scene}: {fixed}/{total} meshes fixed (black -> gray)")

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
