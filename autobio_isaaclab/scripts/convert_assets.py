#!/usr/bin/env python3
"""Batch convert AutoBio MJCF assets to USD for Isaac Lab.

Usage:
    cd /mnt/storage/isaacsim-6.0-official/
    ./isaaclab.sh -p /path/to/autobio_isaaclab/scripts/convert_assets.py

This script:
1. Flattens compositional MJCF scenes
2. Strips custom plugin references (SDF, etc.)
3. Converts to USD via MjcfConverter
4. Outputs to usd_assets/ directory
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add autobio_isaaclab to path
_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_DIR))

# AutoBio model root
_AUTOBIO_ROOT = _PKG_DIR.parent / "autobio"
_MODEL_ROOT = _AUTOBIO_ROOT / "model"
_USD_OUTPUT = _PKG_DIR / "usd_assets"

# Assets to convert
_ROBOT_MODELS = [
    "robot/aloha_left.xml",
]

_OBJECT_MODELS = [
    "object/centrifuge_50ml_screw.xml",
    "object/centrifuge_10slot.xml",
    "object/centrifuge_50ml.xml",
]

_INSTRUMENT_MODELS = [
    "instrument/centrifuge_eppendorf_5430.xml",
    "instrument/thermal_cycler_biorad_c1000.xml",
    "instrument/thermal_mixer_eppendorf_c.xml",
    "instrument/vortex_mixer_genie_2.xml",
]

_SCENE_MODELS = [
    "scene/pickup.xml",
]


def main():
    """Convert all AutoBio MJCF models to USD."""
    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    except ImportError:
        print("ERROR: Must run inside Isaac Lab environment.")
        print("  cd /mnt/storage/isaacsim-6.0-official/")
        print("  ./isaaclab.sh -p scripts/convert_assets.py")
        sys.exit(1)

    from autobio_isaaclab.utils.mjcf_flatten import (
        flatten_mjcf_for_converter,
        create_screw_cap_usd_ready,
    )

    _USD_OUTPUT.mkdir(parents=True, exist_ok=True)

    all_models = _ROBOT_MODELS + _OBJECT_MODELS + _INSTRUMENT_MODELS
    converted = []
    failed = []

    for model_rel in all_models:
        model_path = _MODEL_ROOT / model_rel
        if not model_path.exists():
            print(f"SKIP: {model_path} not found")
            continue

        print(f"\n--- Converting: {model_rel} ---")

        try:
            # For screw-cap model, use simplified version
            if "centrifuge_50ml_screw" in model_rel:
                flat_path = create_screw_cap_usd_ready(model_path, _USD_OUTPUT)
            else:
                flat_path = flatten_mjcf_for_converter(model_path, _USD_OUTPUT)

            # Convert to USD
            usd_name = Path(model_rel).stem
            usd_path = _USD_OUTPUT / usd_name / f"{usd_name}.usd"

            cfg = MjcfConverterCfg(
                asset_path=str(flat_path),
                usd_dir=str(_USD_OUTPUT / usd_name),
                fix_base=True if "robot" in model_rel else False,
                merge_mesh=True,
                collision_from_visuals=False,
                self_collision=True,
            )

            converter = MjcfConverter(cfg)
            print(f"  OK: {usd_path}")
            converted.append(model_rel)

        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(model_rel)

    print(f"\n=== Summary ===")
    print(f"Converted: {len(converted)}/{len(all_models)}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
