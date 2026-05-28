"""Dual-arm configuration for AutoBio Isaac Lab environments.

Extends AutobioEnvCfg for dual-arm tasks: screw_loose, screw_tighten,
pipette, vortex_mixer.

Each arm has 7 DOFs (6 arm joints + 1 gripper), total action/observation
space = 14.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from .autobio_env_cfg import AutobioEnvCfg


@configclass
class AutobioDualEnvCfg(AutobioEnvCfg):
    """Base configuration for dual-arm AutoBio tasks.

    DOF layout (14 total):
      arm1: indices 0-5 (6 joints), gripper1: index 6
      arm2: indices 7-12 (6 joints), gripper2: index 13
    """

    # Spaces -- 7 per arm x 2 arms
    action_space = 14
    observation_space = 14  # joint positions for both arms
    state_space = 0

    # Scene USD path (overridden per-task)
    scene_usd: str = ""

    # Task-specific (overridden by subclasses)
    task_name: str = "dual_base"
