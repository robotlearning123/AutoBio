#!/usr/bin/env python3
"""Fix USD physx.usda files for all AutoBio scenes.

The MuJoCo USD converter generates physx.usda that sublayers physics.usda (Newton).
The Newton physics.usda marks table and robot-base as PhysicsRigidBodyAPI,
which causes PhysX to complain about nested rigid bodies.

This script patches each physx.usda to:
1. Remove PhysicsRigidBodyAPI and PhysicsMassAPI from static geometry (table, base_link parents)
2. Keep PhysicsArticulationRootAPI on the actual articulation root
3. Fix joint body references where needed
"""
from __future__ import annotations

import re
from pathlib import Path

USD_ROOT = Path(__file__).parent.parent / "usd_assets"

# Scenes that need patching (all scenes with a robot on a table)
SCENES = [
    "pickup",
    "insert",
    "mani_thermal_cycler",
    "mani_pipette",
    "lab_screw_all",
    "lab_screw_tighten",
    "insert_centrifuge_5430",
    "mani_thermal_mixer",
    "mani_centrifuge_5430",
    "mani_centrifuge_5910",
    "mani_centrifuge_mini",
    "vortex_mixer",
]

# Per-scene: the static bodies that should NOT be rigid bodies
# These are bodies above the articulation root
STATIC_BODIES = {
    "default": ["table", "aloha1"],  # most scenes: table + aloha1 wrapper
}

# Per-scene: the actual articulation root (first movable link)
ARTICULATION_ROOTS = {
    "default": "left_base_link",
}


def patch_physx_usda(scene_name: str) -> bool:
    """Patch the physx.usda for a scene."""
    # Find the physx.usda
    for ext in [".usda", ".usd", ".usdc"]:
        physx_path = USD_ROOT / scene_name / scene_name / "payloads" / "Physics" / f"physx{ext}"
        if physx_path.exists():
            break
    else:
        print(f"  SKIP: no physx file found for {scene_name}")
        return False

    if not physx_path.name.endswith(".usda"):
        print(f"  SKIP: {physx_path.name} is not text-editable")
        return False

    content = physx_path.read_text()
    original = content

    static_bodies = STATIC_BODIES.get(scene_name, STATIC_BODIES["default"])

    # For each static body, add overrides to remove RigidBodyAPI and MassAPI
    # We insert these before the existing "over" blocks
    patches = []
    for body_name in static_bodies:
        # Check if the body already has an override in the physx.usda
        if f'over "{body_name}"' in content:
            # Add delete directives to the existing override
            # Find the line with the override and add apiSchemas deletion
            pattern = rf'(over "{re.escape(body_name)}")\s*\n(\s*)\{{'
            replacement = rf'\1 (\n\2    delete apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n\2)\n\2{{'
            content = re.sub(pattern, replacement, content, count=1)
        else:
            # Need to add a new override for this body
            # Insert after the "over Geometry {" block
            # Find the right place to insert
            indent = "            "
            patch = f'\n{indent}over "{body_name}" (\n{indent}    delete apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n{indent})\n{indent}{{\n{indent}}}'
            # Find where to insert (after "over Geometry {")
            geo_match = re.search(r'(over "Geometry"\s*\{)', content)
            if geo_match:
                insert_pos = geo_match.end()
                content = content[:insert_pos] + patch + content[insert_pos:]

    if content != original:
        physx_path.write_text(content)
        print(f"  PATCHED: {physx_path}")
        return True
    else:
        print(f"  NO CHANGE: {physx_path}")
        return False


def main():
    print("Patching USD physx variants for PhysX compatibility...")
    patched = 0
    for scene in SCENES:
        print(f"\n{scene}:")
        if patch_physx_usda(scene):
            patched += 1

    print(f"\n{'='*50}")
    print(f"Patched {patched}/{len(SCENES)} scenes")


if __name__ == "__main__":
    main()
