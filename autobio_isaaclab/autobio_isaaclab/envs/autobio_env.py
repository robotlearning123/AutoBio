"""Base AutoBio environment for Isaac Lab.

Provides common functionality for all AutoBio tasks:
- Scene setup with ground plane, table, lights
- Robot articulation management
- Contact-based success checking helpers
- IK and trajectory utilities
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from .autobio_env_cfg import AutobioEnvCfg


class AutobioEnv(DirectRLEnv):
    """Base class for AutoBio task environments."""

    cfg: AutobioEnvCfg

    def __init__(self, cfg: AutobioEnvCfg, render_mode: str | None = None, **kwargs):
        # Allow num_envs override from gym.make()
        if "num_envs" in kwargs:
            cfg.scene.num_envs = kwargs.pop("num_envs")
        super().__init__(cfg, render_mode, **kwargs)

        # Joint indices (populated in subclasses)
        self._arm_dof_indices: list[int] = []
        self._gripper_dof_indices: list[int] = []

        # Cached state
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

    def _setup_scene(self):
        """Set up the common scene elements."""
        # Robot articulation
        self.robot = Articulation(self.cfg.robot_cfg)

        # Ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register robot in scene
        self.scene.articulations["robot"] = self.robot

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store actions for application in _apply_action."""
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        """Apply joint position targets to the robot."""
        # Arm joints: position control
        if self._arm_dof_indices:
            arm_actions = self.actions[:, :len(self._arm_dof_indices)]
            self.robot.set_joint_position_target_index(
                target=arm_actions,
                joint_ids=self._arm_dof_indices,
            )

        # Gripper joints: position control
        if self._gripper_dof_indices:
            gripper_actions = self.actions[:, len(self._arm_dof_indices):]
            self.robot.set_joint_position_target_index(
                target=gripper_actions,
                joint_ids=self._gripper_dof_indices,
            )

    def _get_observations(self) -> dict:
        """Return joint positions as observations."""
        obs = self.joint_pos[:, self._arm_dof_indices + self._gripper_dof_indices]
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Base reward: alive bonus. Subclasses override for task-specific rewards."""
        return torch.ones(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Base done conditions: timeout only. Subclasses add termination."""
        self.joint_pos = self.robot.data.joint_pos.torch
        self.joint_vel = self.robot.data.joint_vel.torch

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset robot to default pose
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim_index(root_pose=default_root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    # --- Helper methods for subclasses ---

    def check_contact(self, geom1_name: str, geom2_name: str) -> torch.Tensor:
        """Check if two geoms are in contact across all envs.

        Uses ContactSensor if available, falls back to RigidObject contact.

        Returns:
            (num_envs,) boolean tensor.
        """
        # This is a simplified check - in practice, use ContactSensor
        # For now, return False (subclasses should implement task-specific checks)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def get_body_pose(self, body_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get body position and orientation.

        Returns:
            Tuple of (pos, quat) each of shape (num_envs, 3) and (num_envs, 4).
        """
        body_idx = self.robot.find_bodies(body_name)[0]
        pos = self.robot.data.body_pos_w[:, body_idx, :] - self.scene.env_origins.unsqueeze(1)
        quat = self.robot.data.body_quat_w[:, body_idx, :]
        return pos.squeeze(1), quat.squeeze(1)
