#!/usr/bin/env python3
"""Post-process USD files to restore original MuJoCo joint/body/site names.

The mujoco_usd_converter mangles prim names: "left/waist" -> "tn__leftwaist_eE"
This script reverses that mangling by reading the original MJCF and building
a mapping from mangled names to original names, then renaming prims in the USD.

Usage:
    cd /mnt/storage/isaacsim-6.0-official/
    ./isaaclab.sh -p /path/to/fix_usd_names.py [--assets aloha_left,insert,...]
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PKG_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PKG_DIR))

_USD_OUTPUT = _PKG_DIR / "usd_assets"
_MODEL_ROOT = _PKG_DIR.parent / "autobio" / "model"

# Pattern: tn__<name_stripped>_<hash>
# name_stripped = original name with "/" and ":" removed
_MANGLED_RE = re.compile(r"^tn__([a-zA-Z0-9_]+?)_([a-zA-Z0-9]{1,4})$")


def _strip_special(name: str) -> str:
    """Strip slashes and colons from a name (what the converter does)."""
    return name.replace("/", "").replace(":", "")


def _build_name_map_from_mjcf(mjcf_path: Path) -> dict[str, str]:
    """Build mapping from stripped-name to original-name from MJCF."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    name_map = {}  # stripped_name -> original_name

    # Collect all named elements: joints, bodies, sites, geoms, cameras, lights
    for elem in root.iter():
        name = elem.get("name")
        if name and ("/" in name or ":" in name):
            stripped = _strip_special(name)
            name_map[stripped] = name

    return name_map


def _build_name_map_from_scene(scene_path: Path) -> dict[str, str]:
    """Build name map from a compositional scene (may reference multiple models)."""
    tree = ET.parse(scene_path)
    root = tree.getroot()

    name_map = {}

    # Collect from all model definitions in <asset><model>
    asset = root.find("asset")
    if asset is not None:
        for model in asset.findall("model"):
            for elem in model.iter():
                name = elem.get("name")
                if name and ("/" in name or ":" in name):
                    stripped = _strip_special(name)
                    name_map[stripped] = name

    # Collect from worldbody (scene-level bodies)
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for elem in worldbody.iter():
            name = elem.get("name")
            if name and ("/" in name or ":" in name):
                stripped = _strip_special(name)
                name_map[stripped] = name

    # Collect from <attach> prefixes
    if worldbody is not None:
        for attach in worldbody.iter("attach"):
            prefix = attach.get("prefix", "")
            model_name = attach.get("model")
            body_name = attach.get("body")
            if prefix and model_name:
                # The prefix will be applied to all body/joint/site names
                # We need to find the model and prefix its names
                if asset is not None:
                    for model in asset.findall("model"):
                        if model.get("name") == model_name:
                            for elem in model.iter():
                                name = elem.get("name")
                                if name:
                                    prefixed = prefix + name
                                    stripped = _strip_special(prefixed)
                                    name_map[stripped] = prefixed

    return name_map


def rename_prims_in_usd(usd_path: Path, name_map: dict[str, str], dry_run: bool = False) -> int:
    """Rename mangled prims in a USD file using the name map.

    Args:
        usd_path: Path to the USD file.
        name_map: Mapping from stripped-name to original-name.
        dry_run: If True, only print what would be renamed.

    Returns:
        Number of renames performed.
    """
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        print(f"  ERROR: Cannot open {usd_path}")
        return 0

    # Find all prims with mangled names
    renames = []  # (old_path, new_name)
    for prim in stage.Traverse():
        name = prim.GetName()
        m = _MANGLED_RE.match(name)
        if m:
            stripped = m.group(1)
            if stripped in name_map:
                original = name_map[stripped]
                # Only rename if the original name is different and contains special chars
                if original != name and ("/" in original or ":" in original):
                    # Use the last component after the last "/" as the prim name
                    # (USD prims can't have "/" in names, so we use the short name)
                    short_name = original.split("/")[-1]
                    renames.append((prim.GetPath(), short_name, original))

    if not renames:
        return 0

    if dry_run:
        for old_path, short, full in renames:
            print(f"  Would rename: {old_path.GetName()} -> {short} (full: {full})")
        return len(renames)

    # Apply renames using Sdf.ChangeBlock for efficiency
    # Process deepest paths first to avoid path conflicts
    renames.sort(key=lambda x: len(str(x[0])), reverse=True)

    edit_layer = stage.GetEditTarget().GetLayer()
    count = 0
    with Sdf.ChangeBlock():
        for old_path, short_name, full_name in renames:
            parent_path = old_path.GetParentPath()
            new_path = parent_path.AppendChild(short_name)

            # Check if new path already exists
            if edit_layer.GetPrimAtPath(new_path):
                continue

            try:
                Sdf.CopySpec(edit_layer, old_path, edit_layer, new_path)
                # Set display name to full original name (e.g., "left/waist")
                new_prim = stage.GetPrimAtPath(new_path)
                if new_prim:
                    new_prim.GetAttribute("displayName")  # Ensure attr exists
                    # Use Sdf to set display name
                    new_spec = edit_layer.GetPrimAtPath(new_path)
                    if new_spec:
                        new_spec.displayName = full_name
                # Remove old prim
                edit_layer.RemovePrim(old_path)
                count += 1
            except Exception as e:
                print(f"  WARN: Failed to rename {old_path} -> {short_name}: {e}")

    if count > 0:
        stage.GetRootLayer().Save()

    return count


def fix_robot_usd(robot_name: str, dry_run: bool = False):
    """Fix joint/body names in a robot USD file."""
    usd_dir = _USD_OUTPUT / robot_name / robot_name
    usd_file = usd_dir / f"{robot_name}.usda"

    if not usd_file.exists():
        print(f"  SKIP: {usd_file} not found")
        return

    # Find the corresponding MJCF
    if robot_name == "aloha_left":
        mjcf = _MODEL_ROOT / "robot" / "aloha_left.xml"
    else:
        print(f"  SKIP: Unknown robot {robot_name}")
        return

    if not mjcf.exists():
        print(f"  SKIP: MJCF {mjcf} not found")
        return

    name_map = _build_name_map_from_mjcf(mjcf)
    print(f"  Built name map with {len(name_map)} entries from {mjcf.name}")

    count = rename_prims_in_usd(usd_file, name_map, dry_run)
    print(f"  Renamed {count} prims in {usd_file.name}")


def fix_scene_usd(scene_name: str, dry_run: bool = False):
    """Fix joint/body names in a scene USD file."""
    usd_dir = _USD_OUTPUT / scene_name / scene_name
    usd_file = usd_dir / f"{scene_name}.usda"

    if not usd_file.exists():
        print(f"  SKIP: {usd_file} not found")
        return

    # Find the corresponding scene MJCF
    mjcf = _MODEL_ROOT / "scene" / f"{scene_name}.xml"
    if not mjcf.exists():
        print(f"  SKIP: MJCF {mjcf} not found")
        return

    name_map = _build_name_map_from_scene(mjcf)
    print(f"  Built name map with {len(name_map)} entries from {mjcf.name}")

    count = rename_prims_in_usd(usd_file, name_map, dry_run)
    print(f"  Renamed {count} prims in {usd_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Fix mangled USD prim names")
    parser.add_argument("--assets", type=str, default=None,
                        help="Comma-separated list of asset names to fix (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed")
    parser.add_argument("--scenes-only", action="store_true", help="Only fix scene USDs")
    parser.add_argument("--robots-only", action="store_true", help="Only fix robot USDs")
    args = parser.parse_args()

    # Robot assets
    robots = ["aloha_left"]
    # Scene assets
    scenes = [
        "pickup", "insert", "mani_thermal_cycler", "mani_pipette",
        "lab_screw_all", "lab_screw_tighten", "insert_centrifuge_5430",
        "mani_thermal_mixer", "mani_centrifuge_5430", "mani_centrifuge_5910",
        "mani_centrifuge_mini", "vortex_mixer",
    ]

    if args.assets:
        requested = set(args.assets.split(","))
        robots = [r for r in robots if r in requested]
        scenes = [s for s in scenes if s in requested]

    if not args.scenes_only:
        print("Fixing robot USDs...")
        for robot in robots:
            print(f"\n--- {robot} ---")
            fix_robot_usd(robot, args.dry_run)

    if not args.robots_only:
        print("\nFixing scene USDs...")
        for scene in scenes:
            print(f"\n--- {scene} ---")
            fix_scene_usd(scene, args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
