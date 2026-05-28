"""Pipette task: aspirate and dispense liquid using a pipette.

Ports AutoBio's ``Pipette`` task to Isaac Lab DirectRLEnv.

Scene: ``usd_assets/mani_pipette/mani_pipette/mani_pipette.usda``

Dual UR5e arms (14 DOF):
  arm1 (dexterous hand): pipette control -- aspirate / move / dispense
  arm2 (2f85 gripper):   tube handling -- pick / position / hold

Success is a 3-phase stateful sequence (placeholder -- returns 0.0 for now).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import sample_uniform

from ..autobio_dual_env import AutobioDualEnv
from ..autobio_dual_env_cfg import AutobioDualEnvCfg

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_PIPETTE_SCENE_USD = os.path.join(_USD_DIR, "mani_pipette", "mani_pipette", "mani_pipette.usda")

# ---------------------------------------------------------------------------
# UR5e joint definitions (dual-arm, 14 DOF total)
# Each arm: 6 arm joints + 1 gripper DOF (simplified from 2-finger to 1-DOF)
# ---------------------------------------------------------------------------

# Arm 1 (UR5e) -- pipette hand
ARM1_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]
GRIPPER1_JOINT_NAMES = [
    "right_driver_joint",
    "left_driver_joint",
    "right_coupler_joint",
    "left_coupler_joint",
    "right_spring_link_joint",
    "left_spring_link_joint",
    "right_follower_joint",
    "left_follower_joint",
]

# Arm 2 (UR5e with _2 suffix) -- tube gripper
ARM2_JOINT_NAMES = [
    "shoulder_pan_2",
    "shoulder_lift_2",
    "elbow_2",
    "wrist_1_2",
    "wrist_2_2",
    "wrist_3_2",
]
# Arm 2 has no separate gripper in this scene — only arm joints
GRIPPER2_JOINT_NAMES: list[str] = []

ALL_JOINT_NAMES = ARM1_JOINT_NAMES + GRIPPER1_JOINT_NAMES + ARM2_JOINT_NAMES + GRIPPER2_JOINT_NAMES

# UR5e stiffness / damping / effort (from ur5e.py)
_UR5E_STIFFNESS = 100.0
_UR5E_DAMPING = 10.0
_UR5E_EFFORT = 150.0  # conservative default; wrist joints are 28 Nm

_GRIPPER_STIFFNESS = 2000.0
_GRIPPER_DAMPING = 100.0
_GRIPPER_EFFORT = 20.0

# Default joint positions (both arms at home pose)
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
    "shoulder_pan_2": 0.0,
    "shoulder_lift_2": -1.5708,
    "elbow_2": 1.5708,
    "wrist_1_2": -1.5708,
    "wrist_2_2": -1.5708,
    "wrist_3_2": 0.0,
}


# ---------------------------------------------------------------------------
# Robot config builder
# ---------------------------------------------------------------------------

def _build_pipette_robot_cfg() -> ArticulationCfg:
    """Build ArticulationCfg for the dual-UR5e pipette scene.

    The scene USD contains both UR5e arms, the pipette, tube rack, and tips.
    """
    actuators = {}

    # Arm 1 joints
    for jname in ARM1_JOINT_NAMES:
        actuators[jname] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=_UR5E_STIFFNESS,
            damping=_UR5E_DAMPING,
            effort_limit=_UR5E_EFFORT,
        )
    # Gripper 1 (Robotiq 2F-85 linkage joints)
    actuators["gripper1"] = ImplicitActuatorCfg(
        joint_names_expr=GRIPPER1_JOINT_NAMES,
        stiffness=_GRIPPER_STIFFNESS,
        damping=_GRIPPER_DAMPING,
        effort_limit=_GRIPPER_EFFORT,
    )

    # Arm 2 joints
    for jname in ARM2_JOINT_NAMES:
        actuators[jname] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=_UR5E_STIFFNESS,
            damping=_UR5E_DAMPING,
            effort_limit=_UR5E_EFFORT,
        )
    # Arm 2 has no separate gripper in this scene

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PIPETTE_SCENE_USD,
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


# Module-level default config
PIPETTE_ROBOT_CFG = _build_pipette_robot_cfg()


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

@configclass
class PipetteEnvCfg(AutobioDualEnvCfg):
    """Configuration for the pipette task."""

    task_name = "pipette"
    episode_length_s = 30.0
    early_stop = True

    # Observation: 20 joint pos + 20 joint vel + 3 tube pos = 43
    observation_space = 43
    action_space = 20  # 6 arm1 + 6 arm2 + 8 gripper

    # Robot: load the pipette scene USD (dual UR5e + pipette + tubes)
    robot_cfg: ArticulationCfg = PIPETTE_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


# ---------------------------------------------------------------------------
# Env implementation
# ---------------------------------------------------------------------------

class PipetteEnv(AutobioDualEnv):
    """Pipette task environment.

    3-phase stateful task (placeholder):
      Phase 1: Arm1 picks up pipette tip
      Phase 2: Arm2 positions tube under pipette
      Phase 3: Arm1 aspirates / dispenses liquid

    Currently returns 0.0 reward (success logic not yet implemented).
    """

    cfg: PipetteEnvCfg

    def __init__(self, cfg: PipetteEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Override DOF indices for this scene's joint order:
        # 0-5: arm1, 6-24: shadow hand (ignored), 25-30: arm2, 31-38: gripper
        self._arm_dof_indices = list(range(6)) + list(range(25, 31))
        self._gripper_dof_indices = list(range(31, 39))

        # Phase tracking (placeholder -- not yet used in reward)
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _setup_scene(self):
        """Set up the pipette scene."""
        super()._setup_scene()

        # Tube body: separate rigid body for tracking
        # USD default prim is tn__pipettemanipulation_wJ, so content is at /Robot/Geometry/...
        tube_prim = self.cfg.robot_cfg.prim_path.replace(
            "/Robot", "/Robot/Geometry/centrifuge_50ml_body/tn__5_centrifuge_50ml_screw_body_"
        )
        self.tube_body = RigidObject(
            RigidObjectCfg(
                prim_path=tube_prim,
                spawn=None,
            )
        )
        self.scene.rigid_objects["tube_body"] = self.tube_body

    def _get_observations(self) -> dict:
        """Return observations: joint pos + joint vel + tip pos + tube pos."""
        # Arm1 (0-5) + arm2 (25-30) + gripper (31-38) = 20 DOF
        dof_indices = self._arm_dof_indices + self._gripper_dof_indices
        joint_pos = self.joint_pos[:, dof_indices]
        joint_vel = self.joint_vel[:, dof_indices]

        # Tube position relative to env origin
        tube_pos = self.tube_body.data.root_pos_w[:, :3] - self.scene.env_origins

        obs = torch.cat([joint_pos, joint_vel, tube_pos], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Placeholder reward: 3-phase success returns 0.0 for now.

        TODO: Implement phase-gated reward:
          Phase 0: tip grasped by arm1 gripper (contact check)
          Phase 1: tube positioned by arm2 (proximity to target pose)
          Phase 2: liquid aspirated / dispensed (volume change sensor)
        """
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Done conditions: timeout + early stop on success."""
        terminated, time_out = super()._get_dones()

        if self.cfg.early_stop:
            reward = self._get_rewards()
            terminated = terminated | (reward > 0)

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset with randomization."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset phase tracking
        self._phase[env_ids] = 0

        # Randomize arm joint positions with small perturbation
        num_reset = len(env_ids)
        num_dof = len(self._arm_dof_indices) + len(self._gripper_dof_indices)
        perturb_low = torch.full((num_dof,), -0.1, device=self.device)
        perturb_high = torch.full((num_dof,), 0.1, device=self.device)
        perturbation = sample_uniform(perturb_low, perturb_high, (num_reset, num_dof), self.device)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        dof_indices = self._arm_dof_indices + self._gripper_dof_indices
        joint_pos[:, dof_indices] += perturbation

        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
