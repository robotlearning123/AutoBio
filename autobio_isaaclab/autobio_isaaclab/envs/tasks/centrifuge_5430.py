"""Centrifuge lid closing task: Eppendorf 5430.

Ports AutoBio's centrifuge lid closing task to Isaac Lab DirectRLEnv.

Scene: ``usd_assets/mani_centrifuge_5430/mani_centrifuge_5430/mani_centrifuge_5430.usda``

The agent controls a single UR5e arm (6 DOF) + Robotiq 2F-85 gripper (2 DOF)
to close the lid of an Eppendorf 5430 centrifuge. The scene USD contains the
full environment (robot + table + centrifuge) as a single articulation.

Joint structure (UR5e + 2F-85, 8 DOF total):
  0-5: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
  6-7: right_driver_joint, left_driver_joint

Centrifuge lid: 1-DOF hinge joint named ``lid``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import sample_uniform

from ..autobio_env import AutobioEnv
from ..autobio_env_cfg import AutobioEnvCfg

if TYPE_CHECKING:
    pass

# Path to the centrifuge scene USDs
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_5430_SCENE_USD = os.path.join(
    _USD_DIR, "mani_centrifuge_5430", "mani_centrifuge_5430", "mani_centrifuge_5430.usda"
)

# UR5e arm joint names (6 DOF)
_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Robotiq 2F-85 gripper joint names (all linkage joints in the USD)
_GRIPPER_JOINT_NAMES = [
    "right_driver_joint",
    "left_driver_joint",
    "right_coupler_joint",
    "left_coupler_joint",
    "right_spring_link_joint",
    "left_spring_link_joint",
    "right_follower_joint",
    "left_follower_joint",
]

# All controllable joint names
_ALL_JOINT_NAMES = _ARM_JOINT_NAMES + _GRIPPER_JOINT_NAMES

# Stiffness per arm joint (kp) -- from UR5e defaults
_ARM_STIFFNESS = {
    "shoulder_pan": 100.0,
    "shoulder_lift": 100.0,
    "elbow": 100.0,
    "wrist_1": 100.0,
    "wrist_2": 100.0,
    "wrist_3": 100.0,
}

# Damping per arm joint (kd)
_ARM_DAMPING = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 10.0,
    "elbow": 10.0,
    "wrist_1": 10.0,
    "wrist_2": 10.0,
    "wrist_3": 10.0,
}

# Effort limits per arm joint (Nm)
_ARM_EFFORT = {
    "shoulder_pan": 150.0,
    "shoulder_lift": 150.0,
    "elbow": 150.0,
    "wrist_1": 28.0,
    "wrist_2": 28.0,
    "wrist_3": 28.0,
}

# Default joint positions (home pose)
_DEFAULT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.5708,
    "elbow": 1.5708,
    "wrist_1": -1.5708,
    "wrist_2": -1.5708,
    "wrist_3": 0.0,
    "right_driver_joint": 0.0,
    "left_driver_joint": 0.0,
    "right_coupler_joint": 0.0,
    "left_coupler_joint": 0.0,
    "right_spring_link_joint": 0.0,
    "left_spring_link_joint": 0.0,
    "right_follower_joint": 0.0,
    "left_follower_joint": 0.0,
}


def _build_centrifuge_robot_cfg(usd_path: str) -> ArticulationCfg:
    """Build ArticulationCfg for a centrifuge scene USD.

    The scene USD contains the full environment (robot + table + centrifuge)
    as a single articulation. The articulation root is the table xform.

    Args:
        usd_path: Path to the scene USD file.

    Returns:
        Configured ArticulationCfg.
    """
    actuators = {}
    for jname in _ARM_JOINT_NAMES:
        actuators[jname] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=_ARM_STIFFNESS[jname],
            damping=_ARM_DAMPING[jname],
            effort_limit=_ARM_EFFORT[jname],
        )
    actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=_GRIPPER_JOINT_NAMES,
        stiffness=2000.0,
        damping=100.0,
        effort_limit=20.0,
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=16,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=_DEFAULT_JOINT_POS,
        ),
        actuators=actuators,
    )


# Default robot configuration for the 5430 scene
CEntrifuge_5430_ROBOT_CFG = _build_centrifuge_robot_cfg(_5430_SCENE_USD)


@configclass
class CentrifugeLidEnvCfg(AutobioEnvCfg):
    """Base configuration for centrifuge lid closing tasks.

    Subclasses override ``scene_usd_path`` and ``episode_length_s`` for each
    specific centrifuge model.
    """

    task_name: str = "centrifuge_lid"
    early_stop: bool = True

    # Observation: 7 joint pos + 7 joint vel + 1 lid joint pos = 15
    observation_space: int = 21
    action_space: int = 14  # 6 arm DOF + 8 gripper linkage joints

    # Lid joint name in the scene articulation
    lid_joint_name: str = "lid_1"

    # Placeholder success -- always returns True (original task used placeholder)
    # Subclasses can override for real success criteria.

    # Randomization ranges for arm joint perturbation at reset
    arm_perturbation_low: tuple[float, ...] = (-0.15, -0.1, -0.1, -0.1, -0.1, -0.1)
    arm_perturbation_high: tuple[float, ...] = (0.15, 0.1, 0.1, 0.1, 0.1, 0.1)

    # Scene USD path -- subclasses must override
    scene_usd_path: str = ""

    # Robot articulation cfg -- subclasses override with scene-specific USD
    robot_cfg: ArticulationCfg = CEntrifuge_5430_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class CentrifugeLidEnv(AutobioEnv):
    """Base environment for centrifuge lid closing tasks.

    The scene USD embeds the robot, table, and centrifuge as a single
    articulation. Joint indices 0-5 are the UR5e arm DOFs and 6-7 are the
    Robotiq 2F-85 gripper drivers. The lid joint is a 1-DOF hinge tracked
    for observations.
    """

    cfg: CentrifugeLidEnvCfg

    def __init__(self, cfg: CentrifugeLidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Arm DOF indices: joints 0-5
        self._arm_dof_indices = list(range(6))

        # Gripper DOF indices: joints 6-13 (8 Robotiq 2F-85 linkage joints)
        self._gripper_dof_indices = list(range(6, 14))

        # Lid joint index -- looked up by name from the articulation
        lid_idx = self.robot.find_joints(self.cfg.lid_joint_name)[0]
        self._lid_dof_index = lid_idx[0] if len(lid_idx) > 0 else None

        # Perturbation tensors for reset randomization
        self._perturb_low = torch.tensor(
            self.cfg.arm_perturbation_low, device=self.device, dtype=torch.float32
        )
        self._perturb_high = torch.tensor(
            self.cfg.arm_perturbation_high, device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        """Set up the centrifuge scene."""
        super()._setup_scene()

    def _get_observations(self) -> dict:
        """Return observations: joint pos + joint vel + lid joint position.

        Observation vector (15-dim):
          [0:6]  arm joint positions
          [6:8]  gripper joint positions
          [8:14] arm joint velocities
          [14]   lid joint position
        """
        arm_pos = self.joint_pos[:, self._arm_dof_indices]
        arm_vel = self.joint_vel[:, self._arm_dof_indices]
        gripper_pos = self.joint_pos[:, self._gripper_dof_indices]

        # Lid joint position
        if self._lid_dof_index is not None:
            lid_pos = self.joint_pos[:, self._lid_dof_index : self._lid_dof_index + 1]
        else:
            lid_pos = torch.zeros(self.num_envs, 1, device=self.device)

        obs = torch.cat([arm_pos, gripper_pos, arm_vel, lid_pos], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Placeholder reward: always 1.0 (success).

        The original AutoBio task used a placeholder success check.
        Override in subclasses for real reward computation.
        """
        return torch.ones(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Done conditions: timeout + early stop on success."""
        terminated, time_out = super()._get_dones()

        if self.cfg.early_stop:
            reward = self._get_rewards()
            terminated = terminated | (reward > 0)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset with arm perturbation randomization."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Randomize arm joint positions with perturbation
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


# --- 5430-specific configuration ---

_5430_ROBOT_CFG = _build_centrifuge_robot_cfg(_5430_SCENE_USD)


@configclass
class Centrifuge5430EnvCfg(CentrifugeLidEnvCfg):
    """Configuration for Eppendorf 5430 centrifuge lid closing task."""

    task_name: str = "centrifuge_5430"
    episode_length_s: float = 15.0

    scene_usd_path: str = _5430_SCENE_USD

    robot_cfg: ArticulationCfg = _5430_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class Centrifuge5430Env(CentrifugeLidEnv):
    """Eppendorf 5430 centrifuge lid closing task.

    15-second episode. Single UR5e + 2F-85 gripper closes the centrifuge lid.
    Success is placeholder (always True).
    """

    cfg: Centrifuge5430EnvCfg

    def __init__(self, cfg: Centrifuge5430EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
