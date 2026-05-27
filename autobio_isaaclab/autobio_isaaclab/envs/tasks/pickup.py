"""Pickup task: pick up a centrifuge tube from a rack.

Ports AutoBio's ``Pickup`` task to Isaac Lab DirectRLEnv.

Original task: ``AutoBio/autobio/pickup_centrifuge_tube.py``
Scene: ``AutoBio/autobio/model/scene/pickup.xml``

The agent controls a single ALOHA arm (6 DOF + gripper) to grasp
a 50ml centrifuge tube from a 10-slot rack.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import sample_uniform

from ..autobio_env import AutobioEnv
from ..autobio_env_cfg import AutobioEnvCfg

if TYPE_CHECKING:
    pass


@configclass
class PickupEnvCfg(AutobioEnvCfg):
    """Configuration for the pickup task."""

    task_name = "pickup"
    episode_length_s = 22.5
    early_stop = True

    # Observation: 7 joint pos + 7 joint vel + 3 tube pos = 17
    observation_space = 17
    action_space = 7  # 6 arm DOF + 1 gripper

    # Success thresholds
    tube_height_threshold: float = 0.925  # z-height for success

    # Randomization ranges
    arm_perturbation_low: tuple[float, ...] = (-0.2, 0.0, 0.0, -0.05, 0.0, -0.1)
    arm_perturbation_high: tuple[float, ...] = (0.2, 0.3, 0.1, 0.05, 0.3, 0.1)

    # Rack grid parameters
    rack_rows: int = 2
    rack_cols: int = 5


class PickupEnv(AutobioEnv):
    """Pickup task environment."""

    cfg: PickupEnvCfg

    def __init__(self, cfg: PickupEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find joint indices
        self._arm_dof_indices, _ = self.robot.find_joints([
            "left/waist", "left/shoulder", "left/elbow",
            "left/forearm_roll", "left/wrist_angle", "left/wrist_rotate",
        ])
        self._gripper_dof_indices, _ = self.robot.find_joints([
            "left/left_finger",
        ])

        # Perturbation tensors
        self._perturb_low = torch.tensor(
            self.cfg.arm_perturbation_low, device=self.device, dtype=torch.float32
        )
        self._perturb_high = torch.tensor(
            self.cfg.arm_perturbation_high, device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        """Set up the pickup scene."""
        super()._setup_scene()

        # TODO: Spawn rack and tube as RigidObject when USD assets are ready
        # For now, the scene only contains the robot
        # self.rack = RigidObject(...)
        # self.tube = RigidObject(...)

    def _get_observations(self) -> dict:
        """Return observations: joint pos + joint vel + tube position."""
        arm_pos = self.joint_pos[:, self._arm_dof_indices]
        arm_vel = self.joint_vel[:, self._arm_dof_indices]
        gripper_pos = self.joint_pos[:, self._gripper_dof_indices]

        # TODO: Add tube position when tube RigidObject is available
        # tube_pos = self.tube.data.root_pos_w[:, :3] - self.scene.env_origins
        tube_pos = torch.zeros(self.num_envs, 3, device=self.device)

        obs = torch.cat([arm_pos, arm_vel, gripper_pos, tube_pos], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Sparse reward: 1.0 if tube is above threshold and gripper contacts tube."""
        # TODO: Implement with actual tube height and contact check
        # tube_height = self.tube.data.root_pos_w[:, 2]
        # contact = self.check_contact("leftfinger", "cap_cyl")
        # success = (tube_height > self.cfg.tube_height_threshold) & contact
        # return success.float()
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Done conditions: timeout + early stop on success."""
        terminated, time_out = super()._get_dones()

        if self.cfg.early_stop:
            # Early stop when task is successful
            reward = self._get_rewards()
            terminated = terminated | (reward > 0)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset with randomization."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Randomize arm joint positions
        num_reset = len(env_ids)
        perturbation = sample_uniform(
            self._perturb_low,
            self._perturb_high,
            (num_reset, len(self._arm_dof_indices)),
            self.device,
        )

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._arm_dof_indices] += perturbation

        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # TODO: Randomize tube position on rack when tube RigidObject is available
        # row = torch.randint(0, self.cfg.rack_rows, (num_reset,), device=self.device)
        # col = torch.randint(0, self.cfg.rack_cols, (num_reset,), device=self.device)
        # tube_pos = self.rack.get_slot_position(row, col)
        # self.tube.write_root_pose_to_sim_index(...)
