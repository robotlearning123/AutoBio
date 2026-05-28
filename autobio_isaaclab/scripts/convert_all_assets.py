#!/usr/bin/env python3
"""Batch convert ALL AutoBio MJCF assets to USD for Isaac Sim 6.

Usage:
    cd /mnt/storage/isaacsim-6.0-official/
    ./isaaclab.sh -p /path/to/convert_all_assets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_DIR))

_AUTOBIO_ROOT = _PKG_DIR.parent / "autobio"
_MODEL_ROOT = _AUTOBIO_ROOT / "model"
_USD_OUTPUT = _PKG_DIR / "usd_assets"


def find_best_xml(model_root: Path, rel_path: str) -> Path | None:
    """Find the best XML file for a model — prefer .gen.xml over .xml."""
    base = model_root / rel_path
    gen = base.with_suffix(".gen.xml")
    if gen.exists():
        return gen
    if base.exists():
        return base
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", help="Skip already converted")
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(argparse.Namespace(headless=True, device="cuda:0"))
    simulation_app = app_launcher.app

    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)

    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    _USD_OUTPUT.mkdir(parents=True, exist_ok=True)

    from autobio_isaaclab.utils.mjcf_flatten import (
        flatten_mjcf_for_converter,
        preprocess_mjcf_for_converter,
        sanitize_mjcf_names,
    )

    # --- Objects: use .gen.xml where available, preprocess .xml ---
    OBJECT_MODELS = [
        "object/centrifuge_50ml_screw.xml",
        "object/centrifuge_10slot.xml",
        "object/centrifuge_50ml.xml",
        "object/centrifuge_1-5ml.xml",
        "object/centrifuge_1-5ml_screw.xml",
        "object/centrifuge_1-5ml_screw-simple.xml",
        "object/centrifuge_15ml.xml",
        "object/centrifuge_15ml_screw.xml",
        "object/centrifuge_10ml.xml",
        "object/centrifuge_1500ul_no_lid.xml",
        "object/pipette_tip.xml",
        "object/pipette_rack.xml",
        "object/pipette.xml",
        "object/pcr_plate_96well.xml",
        "object/cell_dish_100.xml",
        "object/cryovial_1-8ml.xml",
        "object/tip_box.xml",
        "object/centrifuge_plate_60well.xml",
    ]

    # --- Instruments: use raw XML (they use assetdir, convert from original path) ---
    INSTRUMENT_MODELS = [
        "instrument/centrifuge_eppendorf_5430.xml",
        "instrument/centrifuge_eppendorf_5910_ri.xml",
        "instrument/centrifuge_tiangen_tgear_mini.xml",
        "instrument/thermal_cycler_biorad_c1000.xml",
        "instrument/thermal_mixer_eppendorf_c.xml",
        "instrument/vortex_mixer_genie_2.xml",
    ]

    # --- Robot ---
    ROBOT_MODELS = [
        "robot/aloha_left.xml",
    ]

    # --- Scenes: flatten compositional ---
    SCENE_MODELS = [
        "scene/pickup.xml",
        "scene/insert.xml",
        "scene/mani_thermal_cycler.xml",
        "scene/mani_pipette.xml",
        "scene/lab_screw_all.xml",
        "scene/lab_screw_tighten.xml",
        "scene/insert_centrifuge_5430.xml",
        "scene/mani_thermal_mixer.xml",
        "scene/mani_centrifuge_5430.xml",
        "scene/mani_centrifuge_5910.xml",
        "scene/mani_centrifuge_mini.xml",
        "scene/vortex_mixer.xml",
    ]

    converted = []
    failed = []
    skipped = []

    def convert_one(model_rel: str, mjcf_path: Path, import_scene: bool = False):
        name = Path(model_rel).stem
        usd_dir = _USD_OUTPUT / name
        expected_usd = usd_dir / name / f"{name}.usda"

        if args.skip_existing and expected_usd.exists():
            print(f"SKIP (exists): {model_rel}", flush=True)
            skipped.append(model_rel)
            return

        print(f"\n--- {model_rel} ---", flush=True)

        usd_dir.mkdir(parents=True, exist_ok=True)
        config = MJCFImporterConfig(
            mjcf_path=str(mjcf_path.resolve()),
            usd_path=str(usd_dir.resolve()),
            import_scene=import_scene,
            merge_mesh=True,
            collision_from_visuals=False,
            allow_self_collision=True,
        )
        importer = MJCFImporter(config)
        result = importer.import_mjcf()
        print(f"  OK: {result}", flush=True)
        converted.append(model_rel)

    # Convert robots — sanitize names for USD compatibility
    for rel in ROBOT_MODELS:
        path = _MODEL_ROOT / rel
        if not path.exists():
            print(f"SKIP (not found): {rel}", flush=True)
            skipped.append(rel)
            continue
        try:
            path = sanitize_mjcf_names(path, _USD_OUTPUT)
            print(f"  Sanitized: {path.name}", flush=True)
            convert_one(rel, path, import_scene=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
            failed.append((rel, str(e)[:100]))

    # Convert objects — prefer .gen.xml
    for rel in OBJECT_MODELS:
        path = find_best_xml(_MODEL_ROOT, rel)
        if path is None:
            print(f"SKIP (not found): {rel}", flush=True)
            skipped.append(rel)
            continue
        try:
            # Preprocess if it's a plain .xml (not .gen.xml)
            if not path.name.endswith(".gen.xml"):
                path = preprocess_mjcf_for_converter(path, _USD_OUTPUT)
                print(f"  Preprocessed: {path.name}", flush=True)
            convert_one(rel, path)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
            failed.append((rel, str(e)[:100]))

    # Convert instruments — use raw XML from original path
    for rel in INSTRUMENT_MODELS:
        path = _MODEL_ROOT / rel
        if not path.exists():
            print(f"SKIP (not found): {rel}", flush=True)
            skipped.append(rel)
            continue
        try:
            # Only preprocess detent-containing instruments
            raw = path.read_text()
            if "detent" in raw or "mjlab" in raw:
                path = preprocess_mjcf_for_converter(path, _USD_OUTPUT)
                print(f"  Preprocessed: {path.name}", flush=True)
            convert_one(rel, path)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
            failed.append((rel, str(e)[:100]))

    # Convert scenes — flatten compositional
    for rel in SCENE_MODELS:
        path = _MODEL_ROOT / rel
        if not path.exists():
            print(f"SKIP (not found): {rel}", flush=True)
            skipped.append(rel)
            continue
        try:
            flat_path = flatten_mjcf_for_converter(path, _USD_OUTPUT)
            print(f"  Flattened: {flat_path.name}", flush=True)
            convert_one(rel, flat_path, import_scene=True)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
            failed.append((rel, str(e)[:100]))

    print(f"\n{'='*50}", flush=True)
    print(f"Converted: {len(converted)}/{len(ROBOT_MODELS + OBJECT_MODELS + INSTRUMENT_MODELS + SCENE_MODELS)}", flush=True)
    print(f"Skipped: {len(skipped)}", flush=True)
    if failed:
        print(f"Failed ({len(failed)}):", flush=True)
        for name, err in failed:
            print(f"  {name}: {err}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
