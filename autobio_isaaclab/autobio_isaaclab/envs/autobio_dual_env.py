"""Dual-arm AutoBio environment for Isaac Lab.

Extends AutobioEnv with two independent arms:
  arm1: DOF indices 0-5 (joints), 6 (gripper)
  arm2: DOF indices 7-12 (joints), 13 (gripper)

Actions are split equally between the two arms.  Observations include
joint positions for both arms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .autobio_env import AutobioEnv

if TYPE_CHECKING:
    from .autobio_dual_env_cfg import AutobioDualEnvCfg


# DOF index ranges per arm
ARM1_DOF_INDICES = list(range(6))      # joints 0-5
GRIPPER1_DOF_INDEX = [6]               # joint 6
ARM2_DOF_INDICES = list(range(7, 13))  # joints 7-12
GRIPPER2_DOF_INDEX = [13]              # joint 13

ALL_DOF_INDICES = ARM1_DOF_INDICES + GRIPPER1_DOF_INDEX + ARM2_DOF_INDICES + GRIPPER2_DOF_INDEX


class AutobioDualEnv(AutobioEnv):
    """Base class for dual-arm AutoBio task environments."""

    cfg: AutobioDualEnvCfg

    def __init__(self, cfg: AutobioDualEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Override single-arm indices with dual-arm layout
        self._arm1_dof_indices = ARM1_DOF_INDICES
        self._gripper1_dof_index = GRIPPER1_DOF_INDEX
        self._arm2_dof_indices = ARM2_DOF_INDICES
        self._gripper2_dof_index = GRIPPER2_DOF_INDEX

        # Keep parent's _arm_dof_indices / _gripper_dof_indices pointing
        # at the full set so helper methods that use them still work.
        self._arm_dof_indices = ARM1_DOF_INDICES + ARM2_DOF_INDICES
        self._gripper_dof_indices = GRIPPER1_DOF_INDEX + GRIPPER2_DOF_INDEX

    def _apply_action(self) -> None:
        """Split actions between arm1 and arm2, apply position targets."""
        # actions shape: (num_envs, 14)
        arm1_actions = self.actions[:, 0:6]
        gripper1_action = self.actions[:, 6:7]
        arm2_actions = self.actions[:, 7:13]
        gripper2_action = self.actions[:, 13:14]

        # Arm 1 joints
        self.robot.set_joint_position_target_index(
            target=arm1_actions,
            joint_ids=self._arm1_dof_indices,
        )
        # Gripper 1
        self.robot.set_joint_position_target_index(
            target=gripper1_action,
            joint_ids=self._gripper1_dof_index,
        )
        # Arm 2 joints
        self.robot.set_joint_position_target_index(
            target=arm2_actions,
            joint_ids=self._arm2_dof_indices,
        )
        # Gripper 2
        self.robot.set_joint_position_target_index(
            target=gripper2_action,
            joint_ids=self._gripper2_dof_index,
        )

    def _get_observations(self) -> dict:
        """Return joint positions for both arms as observations."""
        obs = self.joint_pos[:, ALL_DOF_INDICES]
        return {"policy": obs}
