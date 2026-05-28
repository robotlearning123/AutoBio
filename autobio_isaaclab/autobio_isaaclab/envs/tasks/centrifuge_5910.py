"""Centrifuge lid closing task: Eppendorf 5910.

Ports AutoBio's centrifuge lid closing task to Isaac Lab DirectRLEnv.

Scene: ``usd_assets/mani_centrifuge_5910/mani_centrifuge_5910/mani_centrifuge_5910.usda``

The agent controls a single UR5e arm (6 DOF) + Robotiq 2F-85 gripper (2 DOF)
to close the lid of an Eppendorf 5910 centrifuge. The 5910 has a larger lid
with a different hinge axis and range compared to the 5430.

Lid joint: ``lid``, hinge, range [0, 1.94] rad, axis (1, 0, 0).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from .centrifuge_5430 import (
    CentrifugeLidEnv,
    CentrifugeLidEnvCfg,
    _build_centrifuge_robot_cfg,
)

# Path to the 5910 scene USD
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_5910_SCENE_USD = os.path.join(
    _USD_DIR, "mani_centrifuge_5910", "mani_centrifuge_5910", "mani_centrifuge_5910.usda"
)

# Robot configuration for the 5910 scene
_5910_ROBOT_CFG = _build_centrifuge_robot_cfg(_5910_SCENE_USD)


@configclass
class Centrifuge5910EnvCfg(CentrifugeLidEnvCfg):
    """Configuration for Eppendorf 5910 centrifuge lid closing task."""

    task_name: str = "centrifuge_5910"
    episode_length_s: float = 30.0

    scene_usd_path: str = _5910_SCENE_USD

    robot_cfg: ArticulationCfg = _5910_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class Centrifuge5910Env(CentrifugeLidEnv):
    """Eppendorf 5910 centrifuge lid closing task.

    30-second episode. Single UR5e + 2F-85 gripper closes the centrifuge lid.
    The 5910 has a larger, heavier lid than the 5430 with a different hinge
    axis. Success is placeholder (always True).
    """

    cfg: Centrifuge5910EnvCfg

    def __init__(self, cfg: Centrifuge5910EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
