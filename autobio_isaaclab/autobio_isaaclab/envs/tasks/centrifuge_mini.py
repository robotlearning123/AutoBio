"""Centrifuge lid closing task: Tiangen TGear Mini.

Ports AutoBio's centrifuge lid closing task to Isaac Lab DirectRLEnv.

Scene: ``usd_assets/mani_centrifuge_mini/mani_centrifuge_mini/mani_centrifuge_mini.usda``

The agent controls a single UR5e arm (6 DOF) + Robotiq 2F-85 gripper (2 DOF)
to close the lid of a Tiangen TGear Mini centrifuge. This is a compact
benchtop centrifuge with a smaller lid.

Lid joint: ``lid``, hinge, range [0, 1.712] rad, axis (-1, 0, 0).
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

# Path to the mini scene USD
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_MINI_SCENE_USD = os.path.join(
    _USD_DIR, "mani_centrifuge_mini", "mani_centrifuge_mini", "mani_centrifuge_mini.usda"
)

# Robot configuration for the mini scene
_MINI_ROBOT_CFG = _build_centrifuge_robot_cfg(_MINI_SCENE_USD)


@configclass
class CentrifugeMiniEnvCfg(CentrifugeLidEnvCfg):
    """Configuration for Tiangen TGear Mini centrifuge lid closing task."""

    task_name: str = "centrifuge_mini"
    episode_length_s: float = 30.0

    scene_usd_path: str = _MINI_SCENE_USD

    robot_cfg: ArticulationCfg = _MINI_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class CentrifugeMiniEnv(CentrifugeLidEnv):
    """Tiangen TGear Mini centrifuge lid closing task.

    30-second episode. Single UR5e + 2F-85 gripper closes the centrifuge lid.
    The mini has a compact lid with a smaller hinge range. Success is
    placeholder (always True).
    """

    cfg: CentrifugeMiniEnvCfg

    def __init__(self, cfg: CentrifugeMiniEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
