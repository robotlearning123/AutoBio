"""AutoBio benchmark environments for Isaac Lab 3.0.

Port of the AutoBio biology lab manipulation benchmark (arXiv 2505.14030)
from MuJoCo to Isaac Sim 6 / Isaac Lab 3.0 with PhysX backend.
"""

import os
import toml

ISAACLAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
ISAACLAB_METADATA = toml.load(os.path.join(ISAACLAB_EXT_DIR, "config", "extension.toml"))
__version__ = ISAACLAB_METADATA["package"]["version"]

# Register Gym environments
import builtins

if not getattr(builtins, "_autobio_isaaclab_registered", False):
    from .utils import import_packages
    import_packages(__name__)
    builtins._autobio_isaaclab_registered = True
