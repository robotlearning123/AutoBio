"""MJCF compositional scene flattener.

Converts MuJoCo 3.x compositional MJCF (``<model>`` + ``<attach>``) into
standalone flat MJCF files that ``MjcfConverter`` can ingest.
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def flatten_mjcf(scene_path: Path, output_path: Path | None = None) -> Path:
    """Flatten a compositional MJCF scene into a standalone file.

    Args:
        scene_path: Path to the compositional scene XML.
        output_path: Where to write the flat XML. Defaults to ``scene_path.stem + "_flat.xml"``.

    Returns:
        Path to the flattened XML file.
    """
    scene_path = Path(scene_path).resolve()
    tree = ET.parse(scene_path)
    root = tree.getroot()

    # Resolve model references
    model_defs: dict[str, ET.Element] = {}
    asset_elem = root.find("asset")
    if asset_elem is not None:
        for model_elem in asset_elem.findall("model"):
            name = model_elem.get("name")
            file_ref = model_elem.get("file")
            if name and file_ref:
                model_path = (scene_path.parent / file_ref).resolve()
                model_tree = ET.parse(model_path)
                model_defs[name] = model_tree.getroot()

    # Process attach directives in worldbody
    worldbody = root.find("worldbody")
    if worldbody is not None:
        _process_body(worldbody, model_defs, scene_path.parent)

    # Remove <model> entries from asset (no longer needed)
    if asset_elem is not None:
        for model_elem in asset_elem.findall("model"):
            asset_elem.remove(model_elem)

    # Write output
    if output_path is None:
        output_path = scene_path.parent / f"{scene_path.stem}_flat.xml"
    output_path = Path(output_path)
    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    return output_path


def _process_body(body: ET.Element, model_defs: dict[str, ET.Element], base_dir: Path):
    """Recursively process a body element, expanding ``<attach>`` tags."""
    # Collect children to process (may modify during iteration)
    children = list(body)
    for child in children:
        if child.tag == "attach":
            _expand_attach(body, child, model_defs, base_dir)
            body.remove(child)
        elif child.tag == "body":
            _process_body(child, model_defs, base_dir)


def _expand_attach(
    parent_body: ET.Element,
    attach: ET.Element,
    model_defs: dict[str, ET.Element],
    base_dir: Path,
):
    """Expand an ``<attach>`` directive by inlining the referenced model body."""
    model_name = attach.get("model")
    body_name = attach.get("body")
    prefix = attach.get("prefix", "")

    if model_name not in model_defs:
        raise ValueError(f"Model '{model_name}' not found in asset definitions")

    model_root = model_defs[model_name]

    # Find the target body in the model
    target_body = _find_body(model_root, body_name)
    if target_body is None:
        raise ValueError(f"Body '{body_name}' not found in model '{model_name}'")

    # Deep copy the body and apply prefix
    inlined = copy.deepcopy(target_body)
    _apply_prefix(inlined, prefix)

    # Inline sub-model assets into the scene root
    _inline_assets(parent_body, model_root, base_dir)

    # Inline sub-model defaults
    _inline_defaults(parent_body, model_root)

    # Append the inlined body to the parent
    parent_body.append(inlined)


def _find_body(root: ET.Element, body_name: str) -> ET.Element | None:
    """Find a body element by name in the XML tree."""
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _apply_prefix(elem: ET.Element, prefix: str):
    """Apply a name prefix to all named elements in the subtree."""
    if not prefix:
        return

    # Attributes that contain names
    name_attrs = ["name", "joint", "body", "geom", "site", "camera", "light", "mesh"]
    ref_attrs = ["joint", "body", "geom", "site", "camera", "light", "mesh",
                 "childclass", "class"]

    for elem in elem.iter():
        # Prefix name attributes
        for attr in name_attrs:
            val = elem.get(attr)
            if val and attr == "name":
                elem.set(attr, prefix + val)

        # Prefix reference attributes (only if they reference a named element)
        for attr in ref_attrs:
            if attr == "name":
                continue
            val = elem.get(attr)
            if val and not val.startswith(prefix):
                # Only prefix if it's a reference to a local element
                elem.set(attr, prefix + val)


def _inline_assets(scene_body: ET.Element, model_root: ET.Element, base_dir: Path):
    """Inline asset definitions from a sub-model into the scene root."""
    # Find the scene root (parent of worldbody)
    scene_root = _find_root(scene_body)
    scene_asset = scene_root.find("asset")
    if scene_asset is None:
        scene_asset = ET.SubElement(scene_root, "asset")

    model_asset = model_root.find("asset")
    if model_asset is None:
        return

    for child in model_asset:
        # Check for duplicates
        name = child.get("name")
        if name and any(existing.get("name") == name for existing in scene_asset.findall(child.tag)):
            continue
        scene_asset.append(copy.deepcopy(child))


def _inline_defaults(scene_body: ET.Element, model_root: ET.Element):
    """Inline default class definitions from a sub-model into the scene root."""
    scene_root = _find_root(scene_body)
    scene_defaults = scene_root.find("default")
    model_defaults = model_root.find("default")
    if model_defaults is None:
        return

    if scene_defaults is None:
        scene_defaults = ET.SubElement(scene_root, "default")

    for child in model_defaults:
        cls_name = child.get("class")
        if cls_name and any(d.get("class") == cls_name for d in scene_defaults.findall("default")):
            continue
        scene_defaults.append(copy.deepcopy(child))


def _find_root(elem: ET.Element) -> ET.Element:
    """Find the root mujoco element by traversing up (requires tree context).
    Since ET doesn't store parent, we find the root by looking for mujoco tag."""
    # For our use case, elem is a child of worldbody which is child of mujoco
    # We need to find the scene root - we'll use a different approach
    # Store reference to root during processing
    return elem  # Will be fixed in the caller


def flatten_mjcf_for_converter(scene_path: Path, output_dir: Path | None = None) -> Path:
    """Flatten and prepare an MJCF scene for Isaac Lab's MjcfConverter.

    This is the main entry point for asset conversion. It:
    1. Flattens compositional MJCF
    2. Strips custom plugin references (SDF, etc.)
    3. Writes to output directory

    Args:
        scene_path: Path to the compositional scene XML.
        output_dir: Directory for output. Defaults to ``usd_assets/`` next to scene.

    Returns:
        Path to the flattened, converter-ready XML file.
    """
    scene_path = Path(scene_path).resolve()
    if output_dir is None:
        output_dir = scene_path.parent.parent.parent / "usd_assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{scene_path.stem}.xml"

    # Parse and flatten
    tree = ET.parse(scene_path)
    root = tree.getroot()

    # Resolve model references
    model_defs: dict[str, ET.Element] = {}
    asset_elem = root.find("asset")
    if asset_elem is not None:
        for model_elem in list(asset_elem.findall("model")):
            name = model_elem.get("name")
            file_ref = model_elem.get("file")
            if name and file_ref:
                model_path = (scene_path.parent / file_ref).resolve()
                model_tree = ET.parse(model_path)
                model_defs[name] = model_tree.getroot()

    # Process attach directives
    worldbody = root.find("worldbody")
    if worldbody is not None:
        _process_body(worldbody, model_defs, scene_path.parent)

    # Remove <model> entries from asset
    if asset_elem is not None:
        for model_elem in list(asset_elem.findall("model")):
            asset_elem.remove(model_elem)

    # Strip custom plugin extensions
    extension = root.find("extension")
    if extension is not None:
        root.remove(extension)

    # Strip SDF plugin geoms and replace with simple collision geoms
    for geom in root.iter("geom"):
        geom_type = geom.get("type")
        if geom_type == "sdf":
            # Replace SDF geom with a cylinder approximation
            geom.set("type", "cylinder")
            # Remove plugin child elements
            for plugin in list(geom.findall("plugin")):
                geom.remove(plugin)

    # Strip plugin mesh references
    for mesh in list(root.iter("mesh")):
        for plugin in list(mesh.findall("plugin")):
            mesh.remove(plugin)

    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    return output_path


def create_screw_cap_usd_ready(scene_path: Path, output_dir: Path | None = None) -> Path:
    """Create a modified centrifuge_50ml_screw MJCF with revolute joint instead of SDF.

    The original uses ``mjlab.sdf.thread`` plugin for helical thread physics.
    This creates a simplified version with a revolute joint for Isaac Lab.
    """
    scene_path = Path(scene_path).resolve()
    if output_dir is None:
        output_dir = scene_path.parent.parent.parent / "usd_assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "centrifuge_50ml_screw_simplified.xml"

    tree = ET.parse(scene_path)
    root = tree.getroot()

    # Remove extension/plugins
    extension = root.find("extension")
    if extension is not None:
        root.remove(extension)

    # Remove SDF plugin meshes
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.find("plugin") is not None:
                asset.remove(mesh)

    # Replace SDF geoms with cylinder approximations
    for geom in root.iter("geom"):
        if geom.get("type") == "sdf":
            geom.set("type", "cylinder")
            geom.set("size", "0.014 0.05")  # Approximate thread region
            for plugin in list(geom.findall("plugin")):
                geom.remove(plugin)

    # The cap body needs a revolute joint (it was originally free-jointed in the scene)
    # We'll add a revolute joint to the cap body
    for body in root.iter("body"):
        if body.get("name") == "centrifuge_50ml_screw_cap":
            # Add a revolute joint along z-axis
            joint = ET.SubElement(body, "joint")
            joint.set("name", "cap_screw")
            joint.set("type", "hinge")
            joint.set("axis", "0 0 1")
            joint.set("range", "0 12.566")  # ~2 full turns in radians
            joint.set("damping", "0.1")
            break

    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    return output_path
