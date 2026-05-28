#!/usr/bin/env python3
"""Convert AutoBio MJCF scenes to PhysX USD via Isaac Sim (single app instance)."""
import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')

import sys
from pathlib import Path

USD_ROOT = Path(__file__).parent.parent / "usd_assets_physx"
SCENE_ROOT = Path(__file__).parent.parent.parent / "autobio" / "model" / "scene"

SCENES = [
    "pickup", "insert", "mani_thermal_cycler", "mani_pipette",
    "lab_screw_all", "lab_screw_tighten", "insert_centrifuge_5430",
    "mani_thermal_mixer", "vortex_mixer", "mani_centrifuge_5430",
    "mani_centrifuge_5910", "mani_centrifuge_mini",
]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else SCENES

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import omni.kit.commands
    from isaacsim.asset.importer.mjcf import _mjcf
    import omni.usd

    USD_ROOT.mkdir(parents=True, exist_ok=True)

    results = {}
    for scene_name in scenes:
        mjcf_path = SCENE_ROOT / f"{scene_name}.xml"
        dest_path = USD_ROOT / f"{scene_name}.usd"

        if not mjcf_path.exists():
            print(f"  SKIP: {scene_name} (no MJCF)")
            results[scene_name] = False
            continue

        print(f"\nConverting: {scene_name}")
        try:
            cfg = _mjcf.ImportConfig()
            cfg.import_inertia_tensor = True
            cfg.fix_base = True
            cfg.make_default_prim = True
            cfg.self_collision = True
            cfg.default_drive_strength = 1000.0
            cfg.density = 1000.0
            cfg.make_instanceable = False
            cfg.merge_fixed_joints = False

            prim_path = f"/World/{scene_name}"

            # Clear stage for next import
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                stage.RemovePrim(prim_path)

            result = omni.kit.commands.execute(
                "MJCFCreateAsset",
                mjcf_path=str(mjcf_path),
                import_config=cfg,
                prim_path=prim_path,
            )

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            stage.Export(str(dest_path))

            if dest_path.exists() and dest_path.stat().st_size > 0:
                sz = dest_path.stat().st_size // 1024
                print(f"  OK: {dest_path.name} ({sz}KB)")
                results[scene_name] = True
            else:
                print(f"  FAIL: empty output")
                results[scene_name] = False

        except Exception as e:
            print(f"  FAIL: {e}")
            results[scene_name] = False

    app.close()

    ok = sum(1 for v in results.values() if v)
    print(f"\nSUMMARY: {ok}/{len(results)} converted")
    for name, success in results.items():
        print(f"  {'OK' if success else 'FAIL'}: {name}")


if __name__ == "__main__":
    main()
