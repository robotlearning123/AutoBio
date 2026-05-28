"""MJCF compositional scene flattener.

Converts MuJoCo 3.x compositional MJCF (``<model>`` + ``<attach>``) into
standalone flat MJCF files that ``MjcfConverter`` / ``MJCFImporter`` can ingest.

Also handles MuJoCo 3.x features unsupported by mujoco_usd_converter:
- Custom plugins (mjlab.sdf.thread, mjlab.passive.detent)
- <replicate> tags (expanded to single instance)
- Trailing colons in mesh references
- childclass references
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def flatten_mjcf_for_converter(scene_path: Path, output_dir: Path | None = None) -> Path:
    """Flatten and prepare an MJCF scene for Isaac Lab's MJCFImporter.

    Args:
        scene_path: Path to the compositional scene XML.
        output_dir: Directory for output.

    Returns:
        Path to the flattened, converter-ready XML file.
    """
    scene_path = Path(scene_path).resolve()
    if output_dir is None:
        output_dir = scene_path.parent.parent.parent / "usd_assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{scene_path.stem}.xml"

    tree = ET.parse(scene_path)
    root = tree.getroot()

    # Step 1: Resolve model definitions from <asset><model .../>
    # Recursively flatten compositional models
    model_defs: dict[str, ET.Element] = {}
    model_dirs: dict[str, Path] = {}
    model_meshdirs: dict[str, Path] = {}
    asset_elem = root.find("asset")
    if asset_elem is not None:
        for model_elem in list(asset_elem.findall("model")):
            name = model_elem.get("name")
            file_ref = model_elem.get("file")
            if name and file_ref:
                model_file_path = (scene_path.parent / file_ref).resolve()
                model_tree = ET.parse(model_file_path)
                model_root_elem = model_tree.getroot()
                model_dirs[name] = model_file_path.parent
                # Compute mesh base dir from model's compiler meshdir/assetdir
                model_compiler = model_root_elem.find("compiler")
                mesh_base = model_file_path.parent
                if model_compiler is not None:
                    for attr in ["meshdir", "assetdir"]:
                        md = model_compiler.get(attr)
                        if md:
                            mesh_base = (model_file_path.parent / md).resolve()
                            break
                model_meshdirs[name] = mesh_base
                # Recursively resolve nested models
                _resolve_nested_models(model_root_elem, model_file_path.parent, model_defs, model_dirs, model_meshdirs)
                model_defs[name] = model_root_elem

    # Step 2: Process <attach> directives in worldbody, inlining into root
    worldbody = root.find("worldbody")
    if worldbody is not None:
        _process_body(root, worldbody, model_defs, model_dirs, model_meshdirs, scene_path.parent)

    # Step 3: Remove <model> entries from asset
    if asset_elem is not None:
        for model_elem in list(asset_elem.findall("model")):
            asset_elem.remove(model_elem)

    # Step 4: Strip custom plugin extensions
    for extension in list(root.findall("extension")):
        root.remove(extension)

    # Step 5: Strip SDF plugin geoms (replace with cylinder approximation)
    for geom in root.iter("geom"):
        if geom.get("type") == "sdf":
            geom.set("type", "cylinder")
            geom.set("size", "0.014 0.05")
            for plugin in list(geom.findall("plugin")):
                geom.remove(plugin)

    # Step 6: Strip plugin mesh children and meshes without file references
    asset_elem2 = root.find("asset")
    removed_mesh_names: set[str] = set()
    if asset_elem2 is not None:
        for mesh in list(asset_elem2.findall("mesh")):
            # Remove plugin children
            for plugin in list(mesh.findall("plugin")):
                mesh.remove(plugin)
            # Remove meshes without file (plugin-generated)
            if not mesh.get("file"):
                removed_mesh_names.add(mesh.get("name", ""))
                asset_elem2.remove(mesh)

    # Strip mesh references to removed meshes
    for geom in root.iter("geom"):
        mesh_ref = geom.get("mesh")
        if mesh_ref in removed_mesh_names:
            del geom.attrib["mesh"]

    # Step 7: Expand <replicate> — keep first instance, remove the tag
    for repl in list(root.iter("replicate")):
        parent = _find_parent(root, repl)
        if parent is None:
            continue
        # Keep the first child of <replicate>, move it to parent
        children = list(repl)
        if children:
            first = copy.deepcopy(children[0])
            # Apply replicate's pos/quat/euler offset if present
            repl_pos = repl.get("pos", "")
            repl_euler = repl.get("euler", "")
            if repl_pos or repl_euler:
                # Merge into child's pos
                _merge_offset(first, repl_pos, repl_euler)
            parent.append(first)
        parent.remove(repl)

    # Step 8: Fix trailing colons in mesh references and normalize mesh names
    # Strip "obj:" prefixes from mesh names (MuJoCo 3.x convention)
    # Generate mesh names from file paths when missing (MuJoCo auto-generates)
    asset_elem = root.find("asset")
    if asset_elem is not None:
        for mesh in asset_elem.findall("mesh"):
            name = mesh.get("name")
            if name and ":" in name:
                mesh.set("name", name.split(":")[-1])
            elif not name:
                # Generate name from file path (MuJoCo convention)
                file_attr = mesh.get("file", "")
                if file_attr:
                    stem = Path(file_attr).stem
                    mesh.set("name", stem)

    for elem in root.iter():
        val = elem.get("mesh")
        if val:
            # Strip trailing colon (MuJoCo 3.x texture separator)
            val = val.rstrip(":")
            # Strip prefix (e.g. "obj:foo" → "foo")
            if ":" in val:
                val = val.split(":")[-1]
            elem.set("mesh", val)

    # Step 9: Strip any remaining class/childclass refs (resolution happened per-model).
    for elem in root.iter():
        if elem.tag == "default":
            continue
        for attr in ("class", "childclass"):
            if elem.get(attr):
                del elem.attrib[attr]

    # Step 10: Remove <plugin> elements anywhere
    for plugin in list(root.iter("plugin")):
        parent = _find_parent(root, plugin)
        if parent is not None:
            parent.remove(plugin)

    # Step 11: Fix meshdir/assetdir to use absolute path
    # For compositional scenes, all models share the same assets root
    assets_root = (scene_path.parent.parent.parent / "assets").resolve()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
        # Insert compiler as first child
        root.insert(0, compiler)
    for attr in ["meshdir", "assetdir"]:
        val = compiler.get(attr)
        if val:
            abs_val = (scene_path.parent / val).resolve()
            compiler.set(attr, str(abs_val))
        else:
            # Set default assets directory
            compiler.set(attr, str(assets_root))

    # Step 12: Sanitize names for USD compatibility.
    # The mujoco_usd_converter mangles names containing "/" or ":" (e.g., "left/waist"
    # becomes "tn__leftwaist_eE"). Replace "/" with "_" so the converter produces clean names.
    _sanitize_names_for_usd(root)

    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    return output_path


def _find_parent(root: ET.Element, target: ET.Element) -> ET.Element | None:
    """Find the parent of target element."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None


def _merge_offset(elem: ET.Element, pos_str: str, euler_str: str):
    """Merge replicate offset into element's pos attribute."""
    existing_pos = [0.0, 0.0, 0.0]
    if elem.get("pos"):
        existing_pos = [float(x) for x in elem.get("pos").split()]
    if pos_str:
        offset = [float(x) for x in pos_str.split()]
        existing_pos = [a + b for a, b in zip(existing_pos, offset)]
    elem.set("pos", " ".join(str(x) for x in existing_pos))


def _resolve_nested_models(
    model_root: ET.Element,
    model_dir: Path,
    all_defs: dict[str, ET.Element],
    all_dirs: dict[str, Path],
    all_meshdirs: dict[str, Path],
):
    """Recursively resolve nested model definitions within a model."""
    asset = model_root.find("asset")
    if asset is None:
        return
    for model_elem in list(asset.findall("model")):
        name = model_elem.get("name")
        file_ref = model_elem.get("file")
        if name and file_ref:
            if name in all_defs:
                continue
            model_file_path = (model_dir / file_ref).resolve()
            model_tree = ET.parse(model_file_path)
            sub_root = model_tree.getroot()
            all_dirs[name] = model_file_path.parent
            # Compute mesh base dir
            model_compiler = sub_root.find("compiler")
            mesh_base = model_file_path.parent
            if model_compiler is not None:
                for attr in ["meshdir", "assetdir"]:
                    md = model_compiler.get(attr)
                    if md:
                        mesh_base = (model_file_path.parent / md).resolve()
                        break
            all_meshdirs[name] = mesh_base
            # Recurse
            _resolve_nested_models(sub_root, model_file_path.parent, all_defs, all_dirs, all_meshdirs)
            all_defs[name] = sub_root


def _process_body(
    scene_root: ET.Element,
    body: ET.Element,
    model_defs: dict[str, ET.Element],
    model_dirs: dict[str, Path],
    model_meshdirs: dict[str, Path],
    base_dir: Path,
):
    """Recursively process body/frame elements, expanding <attach> tags."""
    for child in list(body):
        if child.tag == "attach":
            _expand_attach(scene_root, body, child, model_defs, model_dirs, model_meshdirs, base_dir)
            body.remove(child)
        elif child.tag in ("body", "frame"):
            _process_body(scene_root, child, model_defs, model_dirs, model_meshdirs, base_dir)


def _expand_attach(
    scene_root: ET.Element,
    parent_body: ET.Element,
    attach: ET.Element,
    model_defs: dict[str, ET.Element],
    model_dirs: dict[str, Path],
    model_meshdirs: dict[str, Path],
    base_dir: Path,
):
    """Expand an <attach> directive by inlining the referenced model body."""
    model_name = attach.get("model")
    body_name = attach.get("body")
    prefix = attach.get("prefix", "")

    if model_name not in model_defs:
        raise ValueError(f"Model '{model_name}' not found")

    model_root = model_defs[model_name]
    target_body = _find_body(model_root, body_name)
    if target_body is None:
        raise ValueError(f"Body '{body_name}' not found in model '{model_name}'")

    # Deep copy
    inlined = copy.deepcopy(target_body)

    # Inline assets into scene root
    mesh_base = model_meshdirs.get(model_name, base_dir)
    _inline_into_root(scene_root, model_root, "asset", model_base_dir=mesh_base, scene_base_dir=base_dir)

    # Inline defaults with class prefix to avoid conflicts between models
    class_prefix = model_name + "_"
    _inline_into_root(scene_root, model_root, "default", class_prefix=class_prefix)

    # Resolve class/childclass in the body tree using the MODEL's defaults
    # (before prefixing). This merges default attributes (type, size, etc.)
    # into elements and strips class/childclass refs.
    model_default_tree = _build_default_tree(model_root)
    _propagate_childclass_in_tree(inlined, default_tree=model_default_tree)
    _resolve_class_defaults_in_tree(inlined, model_default_tree)

    # Recursively expand nested <attach> tags
    _process_body(scene_root, inlined, model_defs, model_dirs, model_meshdirs, base_dir)

    # Apply prefix to body names/references AFTER nested expansion
    _apply_prefix(inlined, prefix)

    parent_body.append(inlined)


def _find_body(root: ET.Element, body_name: str) -> ET.Element | None:
    if body_name == "world":
        # Special case: "world" means the worldbody itself — wrap its children
        # in a virtual body so they can be inlined
        worldbody = root.find("worldbody")
        if worldbody is not None:
            wrapper = ET.Element("body", name="__world_wrapper__")
            for child in list(worldbody):
                wrapper.append(child)
            return wrapper
        return None
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _apply_prefix(elem: ET.Element, prefix: str):
    """Apply name prefix to all named elements and references.

    Note: class/childclass are NOT prefixed — they reference shared <default>
    definitions that are inlined once into the scene root.
    """
    if not prefix:
        return

    for sub in elem.iter():
        # Prefix name attributes
        name = sub.get("name")
        if name:
            sub.set("name", prefix + name)

        # Prefix references to local elements
        for attr in ["joint", "geom", "site", "camera", "light", "body",
                      "geom1", "geom2", "body1", "body2", "joint1", "joint2"]:
            ref = sub.get(attr)
            if ref and not ref.startswith(prefix):
                sub.set(attr, prefix + ref)


def _inline_into_root(
    scene_root: ET.Element,
    model_root: ET.Element,
    tag: str,
    model_base_dir: Path | None = None,
    scene_base_dir: Path | None = None,
    class_prefix: str = "",
    ref_prefix: str = "",
):
    """Inline child elements from model_root's <tag> into scene_root's <tag>.

    When class_prefix is provided and tag is "default", all class names in the
    inlined defaults are prefixed to avoid collisions between models.
    When ref_prefix is provided, geom/body/joint references in inlined elements
    are prefixed (for contact/equality pairs).
    """
    scene_section = scene_root.find(tag)
    model_section = model_root.find(tag)
    if model_section is None:
        return

    if scene_section is None:
        scene_section = ET.SubElement(scene_root, tag)

    for child in model_section:
        # Rebuild dedup state from current scene_section contents each iteration,
        # so multiple _inline_into_root calls for the same section see each other's additions.
        existing_tag_names: set[tuple[str, str]] = set()
        existing_mesh_files: set[str] = set()
        for elem in scene_section:
            name = elem.get("name") or elem.get("class")
            if name:
                existing_tag_names.add((elem.tag, name))
            if elem.tag == "mesh" and elem.get("file"):
                existing_mesh_files.add(elem.get("file"))

        # Generate mesh names from file paths if missing (MuJoCo auto-generates these)
        if child.tag == "mesh" and not child.get("name") and child.get("file"):
            stem = Path(child.get("file")).stem
            child.set("name", stem)

        # For assets, use name only (not class) as identifier — many asset
        # elements share the same class (e.g., all 2f85 meshes have class="2f85")
        if tag == "asset":
            child_id = child.get("name")
        else:
            child_id = child.get("name") or child.get("class")
        # Apply class prefix to default class names to avoid collisions
        if class_prefix and tag == "default":
            if child_id:
                child_id = class_prefix + child_id
        # Skip if same (tag, name) already exists
        if child_id and (child.tag, child_id) in existing_tag_names:
            continue
        # For unnamed assets (meshes, textures), dedup by file path.
        # Must resolve relative paths to absolute to match already-inlined meshes.
        if not child_id and child.get("file"):
            file_attr = child.get("file", "")
            if file_attr:
                if model_base_dir and not Path(file_attr).is_absolute():
                    abs_file = str((model_base_dir / file_attr).resolve())
                else:
                    abs_file = file_attr
                if abs_file in existing_mesh_files:
                    continue

        inlined = copy.deepcopy(child)
        # Apply class prefix to the inlined default element
        if class_prefix and tag == "default":
            _prefix_default_class(inlined, class_prefix)
        # Apply ref prefix to geom/body/joint references in contact/equality
        if ref_prefix:
            _apply_prefix(inlined, ref_prefix)
        # Fix mesh file paths if we know the model's base directory
        if model_base_dir and scene_base_dir:
            _fix_mesh_paths(inlined, model_base_dir, scene_base_dir)
        scene_section.append(inlined)


def _resolve_all_class_defaults(root: ET.Element):
    """Resolve class/childclass references by inlining default attributes.

    MuJoCo class inheritance works like CSS: a <default class="visual"> defines
    attributes that all elements with class="visual" inherit. We must inline
    these before stripping class refs, or elements lose critical attributes
    (size, type, friction, etc.).
    """
    # Build default class tree
    default_tree = _build_default_tree(root)

    # Propagate childclass to all descendants of bodies that set it
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody:
            if body.tag in ("body", "frame"):
                _propagate_childclass_in_tree(body, default_tree=default_tree)

    # For each non-default element with a class attribute, merge defaults
    for elem in root.iter():
        if elem.tag == "default":
            continue
        cls = elem.get("class")
        if cls:
            _merge_class_defaults(elem, cls, default_tree, elem.tag)
        # Strip class/childclass after resolving
        for attr in ("class", "childclass"):
            if elem.get(attr):
                del elem.attrib[attr]


def _resolve_class_defaults_in_tree(body: ET.Element, default_tree: dict):
    """Resolve class defaults in a body tree (called per-model during attach)."""
    for elem in body.iter():
        if elem.tag == "default":
            continue
        cls = elem.get("class")
        if cls:
            _merge_class_defaults(elem, cls, default_tree, elem.tag)
        # Strip class/childclass after resolving
        for attr in ("class", "childclass"):
            if elem.get(attr):
                del elem.attrib[attr]


def _build_default_tree(root: ET.Element) -> dict:
    """Build a tree of default class definitions for lookup.

    Returns dict: class_name → {"attrs": {tag: {attr: val}},
                                 "parent": parent_class_name_or_None}
    """
    tree: dict[str, dict] = {}
    default_elem = root.find("default")
    if default_elem is None:
        return tree

    def walk_defaults(default_section, parent_class=None):
        for child in default_section:
            if child.tag != "default":
                continue
            cls = child.get("class")
            if not cls:
                # Anonymous default — skip (it's just a container)
                walk_defaults(child, parent_class)
                continue

            # Build full class path
            if parent_class:
                full_cls = parent_class + "/" + cls
            else:
                full_cls = cls

            # Collect attributes from this default (skip 'class')
            # Group by child element tag (e.g., <geom type="..." size="...">)
            attrs: dict[str, dict[str, str]] = {}
            for attr_name, attr_val in child.attrib.items():
                if attr_name == "class":
                    continue
                # Top-level attributes on <default> itself apply to same-tag element
                attrs.setdefault(child.tag, {})[attr_name] = attr_val

            # Also collect child element attributes (e.g., <geom .../> inside <default>)
            for sub_child in child:
                if sub_child.tag == "default":
                    continue
                tag = sub_child.tag
                for attr_name, attr_val in sub_child.attrib.items():
                    attrs.setdefault(tag, {})[attr_name] = attr_val

            tree[full_cls] = {"attrs": attrs, "parent": parent_class}

            # Recurse into nested defaults
            walk_defaults(child, full_cls)

    walk_defaults(default_elem)
    return tree


def _resolve_class_in_context(cls: str, context: str, tree: dict | None) -> str:
    """Resolve a class name within a childclass context.

    In MuJoCo, class="sphere_collision" under childclass="vx300s" searches
    the vx300s default's subtree for sphere_collision. The result is the
    full path like "vx300s/collision/sphere_collision".
    """
    if tree is None:
        return context + "/" + cls

    # First try direct match: context + "/" + cls
    direct = context + "/" + cls
    if direct in tree:
        return direct

    # Search deeper: look for context/*/.../cls in the tree
    prefix = context + "/"
    for key in tree:
        if key.startswith(prefix) and key.endswith("/" + cls):
            return key

    # Fallback: just prepend context
    return direct


def _merge_class_defaults(elem: ET.Element, cls: str, tree: dict, elem_tag: str):
    """Merge default attributes into elem for class cls.

    Walks up the class hierarchy, applying attributes from each level.
    More specific defaults (leaf) override less specific (root).
    Only sets attributes that elem doesn't already have.
    """
    # Collect the full inheritance chain
    chain = []
    current = cls
    while current and current in tree:
        chain.append(tree[current])
        current = tree[current]["parent"]

    # Apply from ROOT down, but DON'T skip — later (more specific) entries
    # must override earlier (less specific) ones.
    # First pass: collect all attrs from root to leaf
    merged: dict[str, str] = {}
    for entry in chain[::-1]:  # root first
        tag_attrs = entry["attrs"].get(elem_tag, {})
        merged.update(tag_attrs)

    # Second pass: set on elem (only missing attrs)
    for attr, val in merged.items():
        if elem.get(attr) is None:
            elem.set(attr, val)


def _propagate_childclass_in_tree(
    body: ET.Element,
    parent_childclass: str | None = None,
    default_tree: dict | None = None,
):
    """Resolve childclass by finding the full class path in the default hierarchy.

    In MuJoCo, childclass on a body changes how class references are resolved:
    class="sphere_collision" under childclass="vx300s" resolves to
    "vx300s/collision/sphere_collision" (searching through the hierarchy).
    Elements with no class reference are NOT affected.
    """
    cc = body.get("childclass") or parent_childclass
    # Strip childclass from body itself
    if body.get("childclass"):
        del body.attrib["childclass"]
    for child in body:
        if child.tag in ("body", "frame"):
            _propagate_childclass_in_tree(child, cc, default_tree)
        elif child.tag in ("geom", "joint", "site", "camera", "light",
                           "position", "velocity", "actuator",
                           "motor", "position", "velocity"):
            cls = child.get("class")
            if cls and cc:
                # Resolve class within childclass context
                resolved = _resolve_class_in_context(cls, cc, default_tree)
                child.set("class", resolved)


def _prefix_default_class(elem: ET.Element, prefix: str):
    """Prefix the class attribute on a <default> element and all nested default children.

    E.g., <default class="visual"> → <default class="ur5e_visual">
    Also handles nested <default class="visual/grasp"> → <default class="ur5e_visual/grasp">
    """
    cls = elem.get("class")
    if cls:
        # Handle slash-separated class paths: prefix the root segment
        parts = cls.split("/")
        parts[0] = prefix + parts[0]
        elem.set("class", "/".join(parts))
    for child in elem:
        if child.tag == "default":
            _prefix_default_class(child, prefix)


def _apply_class_prefix(elem: ET.Element, prefix: str):
    """Resolve class/childclass references by inlining default attributes.

    MuJoCo's class inheritance provides default attribute values to elements.
    We can't just strip class refs — elements need those defaults. Instead,
    look up the default for each element's class and merge missing attributes.
    Then strip the class/childclass ref.
    """
    # This is called on body trees, not the full root, so we can't resolve
    # defaults here. Instead, just strip — the global Step 9 handles it.
    pass


def _fix_mesh_paths(elem: ET.Element, model_meshdir: Path, scene_dir: Path):
    """Fix mesh file paths to absolute (resolved from model's meshdir)."""
    if elem.tag == "mesh":
        file_attr = elem.get("file")
        if file_attr and not Path(file_attr).is_absolute():
            # Resolve relative to model's meshdir (assets root)
            abs_path = (model_meshdir / file_attr).resolve()
            elem.set("file", str(abs_path))


def _sanitize_names_for_usd(root: ET.Element):
    """Replace special characters in element names for USD compatibility.

    The mujoco_usd_converter generates mangled prim names for elements with "/" or ":"
    or "-" in their names (e.g., "left/waist" -> "tn__leftwaist_eE", "lid-lever" -> "tn__lidlever_...").
    By replacing these characters before conversion, we get clean prim names.

    Replacements:
    - ":" prefix stripped (e.g., "obj:foo" -> "foo")
    - "/" -> "_" (e.g., "left/waist" -> "left_waist")
    - "-" -> "_" (e.g., "lid-lever" -> "lid_lever")
    """
    # Collect all named elements and build rename map.
    # Track used names per element tag to avoid false conflicts (MuJoCo allows
    # same name across different element types, e.g., body "left/waist" and joint "left/waist").
    name_map = {}  # old_name -> new_name
    used_names_by_tag: dict[str, set[str]] = {}  # tag -> set of names

    # First pass: collect all existing names by tag
    for elem in root.iter():
        name = elem.get("name")
        if name:
            used_names_by_tag.setdefault(elem.tag, set()).add(name)

    # Second pass: build rename map
    for elem in root.iter():
        name = elem.get("name")
        if not name:
            continue
        new_name = name
        # Strip ":" prefix (e.g., "obj:foo" -> "foo")
        if ":" in new_name:
            new_name = new_name.split(":")[-1]
        # Replace "/" and "-" with "_" (both cause USD name mangling)
        for ch in ("/", "-"):
            if ch in new_name:
                new_name = new_name.replace(ch, "_")
        if new_name != name:
            tag = elem.tag
            used = used_names_by_tag.setdefault(tag, set())
            # Check for conflicts within the same element type
            if new_name in used and new_name not in name_map:
                counter = 2
                while f"{new_name}_{counter}" in used:
                    counter += 1
                new_name = f"{new_name}_{counter}"
            name_map[name] = new_name
            used.add(new_name)

    if not name_map:
        return

    # Apply renames to all name attributes
    for elem in root.iter():
        name = elem.get("name")
        if name and name in name_map:
            elem.set("name", name_map[name])

    # Update all reference attributes that point to named elements
    ref_attrs = [
        "joint", "body", "site", "camera", "light", "mesh",
        "geom", "material", "texture", "class", "childclass",
        "geom1", "geom2", "body1", "body2", "joint1", "joint2",
        "target",
    ]
    for elem in root.iter():
        for attr in ref_attrs:
            val = elem.get(attr)
            if val and val in name_map:
                elem.set(attr, name_map[val])

    # Update class names in <default> elements
    for default in root.iter("default"):
        cls = default.get("class")
        if cls and cls in name_map:
            default.set("class", name_map[cls])


def preprocess_mjcf_for_converter(model_path: Path, output_dir: Path | None = None) -> Path:
    """Preprocess a standalone (non-compositional) MJCF for mujoco_usd_converter.

    Handles MuJoCo 3.x features that the converter doesn't support:
    - Custom plugins (mjlab.sdf.thread, mjlab.passive.detent)
    - <replicate> tags
    - Trailing colons in mesh references
    - SDF geoms replaced with cylinder approximations
    - Detent geoms removed

    Args:
        model_path: Path to the MJCF file.
        output_dir: Directory for output (defaults to usd_assets/).

    Returns:
        Path to the preprocessed XML file.
    """
    model_path = Path(model_path).resolve()
    if output_dir is None:
        output_dir = model_path.parent.parent.parent / "usd_assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_path.stem}.xml"

    # Fix malformed XML declarations (e.g., "versi on" typo)
    raw = model_path.read_text(encoding="utf-8")
    raw = raw.replace("versi on=", "version=")
    root = ET.fromstring(raw)
    tree = ET.ElementTree(root)
    root = tree.getroot()

    # Strip <extension> elements
    for ext in list(root.findall("extension")):
        root.remove(ext)

    # Strip SDF plugin meshes and meshes without file from <asset>
    asset = root.find("asset")
    removed_mesh_names: set[str] = set()
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.find("plugin") is not None or not mesh.get("file"):
                removed_mesh_names.add(mesh.get("name", ""))
                asset.remove(mesh)

    # Strip mesh references to removed meshes
    for geom in root.iter("geom"):
        mesh_ref = geom.get("mesh")
        if mesh_ref in removed_mesh_names:
            del geom.attrib["mesh"]

    # Replace SDF geoms with cylinder approximations
    for geom in root.iter("geom"):
        if geom.get("type") == "sdf":
            geom.set("type", "cylinder")
            geom.set("size", "0.014 0.05")
            for plugin in list(geom.findall("plugin")):
                geom.remove(plugin)

    # Remove detent plugin references from geoms
    for geom in root.iter("geom"):
        for plugin in list(geom.findall("plugin")):
            if "detent" in (plugin.get("plugin", "") or "").lower():
                geom.remove(plugin)

    # Expand <replicate> — keep first instance
    for repl in list(root.iter("replicate")):
        parent = _find_parent(root, repl)
        if parent is None:
            continue
        children = list(repl)
        if children:
            first = copy.deepcopy(children[0])
            repl_pos = repl.get("pos", "")
            repl_euler = repl.get("euler", "")
            if repl_pos or repl_euler:
                _merge_offset(first, repl_pos, repl_euler)
            parent.append(first)
        parent.remove(repl)

    # Fix trailing colons in mesh references and normalize mesh names
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = mesh.get("name", "")
            if ":" in name:
                mesh.set("name", name.split(":")[-1])

    for elem in root.iter():
        val = elem.get("mesh")
        if val:
            val = val.rstrip(":")
            if ":" in val:
                val = val.split(":")[-1]
            elem.set("mesh", val)

    # Remove <plugin> elements anywhere
    for plugin in list(root.iter("plugin")):
        parent = _find_parent(root, plugin)
        if parent is not None:
            parent.remove(plugin)

    # Fix meshdir/assetdir to use absolute path (since we're writing to a different directory)
    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ["meshdir", "assetdir"]:
            val = compiler.get(attr)
            if val:
                abs_val = (model_path.parent / val).resolve()
                compiler.set(attr, str(abs_val))

    # Sanitize names for USD compatibility (replace "/" with "_")
    _sanitize_names_for_usd(root)

    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    return output_path


def sanitize_mjcf_names(model_path: Path, output_dir: Path | None = None) -> Path:
    """Sanitize names in a standalone MJCF for USD compatibility.

    Lightweight version of preprocess_mjcf_for_converter that only replaces "/" with "_"
    in element names, without stripping plugins/extensions or modifying geometry.
    Suitable for robot MJCFs that don't need heavy preprocessing.

    Args:
        model_path: Path to the MJCF file.
        output_dir: Directory for output (defaults to usd_assets/).

    Returns:
        Path to the sanitized XML file.
    """
    model_path = Path(model_path).resolve()
    if output_dir is None:
        output_dir = model_path.parent.parent.parent / "usd_assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_path.stem}.xml"

    raw = model_path.read_text(encoding="utf-8")
    raw = raw.replace("versi on=", "version=")
    root = ET.fromstring(raw)
    tree = ET.ElementTree(root)

    # Fix meshdir/assetdir to use absolute path
    compiler = root.find("compiler")
    if compiler is not None:
        for attr in ["meshdir", "assetdir"]:
            val = compiler.get(attr)
            if val:
                abs_val = (model_path.parent / val).resolve()
                compiler.set(attr, str(abs_val))

    # Sanitize names for USD compatibility (replace "/" with "_")
    _sanitize_names_for_usd(root)

    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    return output_path


def create_screw_cap_usd_ready(scene_path: Path, output_dir: Path | None = None) -> Path:
    """Create a simplified centrifuge_50ml_screw MJCF with revolute joint instead of SDF."""
    scene_path = Path(scene_path).resolve()
    if output_dir is None:
        output_dir = scene_path.parent.parent.parent / "usd_assets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "centrifuge_50ml_screw_simplified.xml"

    tree = ET.parse(scene_path)
    root = tree.getroot()

    # Remove extension/plugins
    for ext in list(root.findall("extension")):
        root.remove(ext)

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
            geom.set("size", "0.014 0.05")
            for plugin in list(geom.findall("plugin")):
                geom.remove(plugin)

    # Add revolute joint to cap body
    for body in root.iter("body"):
        if body.get("name") == "centrifuge_50ml_screw_cap":
            joint = ET.SubElement(body, "joint")
            joint.set("name", "cap_screw")
            joint.set("type", "hinge")
            joint.set("axis", "0 0 1")
            joint.set("range", "0 12.566")
            joint.set("damping", "0.1")
            break

    tree.write(output_path, xml_declaration=True, encoding="utf-8")
    return output_path
