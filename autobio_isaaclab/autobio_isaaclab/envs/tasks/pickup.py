"""Pickup task: pick up a centrifuge tube from a rack.

Ports AutoBio's ``Pickup`` task to Isaac Lab DirectRLEnv.

Original task: ``AutoBio/autobio/pickup_centrifuge_tube.py``
Scene: ``usd_assets/pickup/pickup/pickup.usda``

The agent controls a single ALOHA arm (6 DOF + gripper) to grasp
a 50ml centrifuge tube from a 10-slot rack.

Scene USD contains the full environment (robot + table + rack + tube)
as a single articulation. The tube body and cap are separate rigid bodies
tracked via RigidObject.
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

from ..autobio_env import AutobioEnv
from ..autobio_env_cfg import AutobioEnvCfg

if TYPE_CHECKING:
    pass

# Path to the pickup scene USD
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_PICKUP_SCENE_USD = os.path.join(_USD_DIR, "pickup", "pickup", "pickup.usda")

# Joint names in the order they appear in the articulation
_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_left_finger",
    "left_right_finger",
]

# Stiffness per joint (kp)
_STIFFNESS = {
    "left_waist": 43.0,
    "left_shoulder": 265.0,
    "left_elbow": 227.0,
    "left_forearm_roll": 78.0,
    "left_wrist_angle": 37.0,
    "left_wrist_rotate": 10.4,
    "left_left_finger": 2000.0,
    "left_right_finger": 2000.0,
}

# Damping per joint (kd)
_DAMPING = {
    "left_waist": 5.76,
    "left_shoulder": 20.0,
    "left_elbow": 18.49,
    "left_forearm_roll": 6.78,
    "left_wrist_angle": 6.28,
    "left_wrist_rotate": 1.2,
    "left_left_finger": 124.0,
    "left_right_finger": 124.0,
}

# Effort limits per joint
_EFFORT = {
    "left_waist": 35.0,
    "left_shoulder": 144.0,
    "left_elbow": 59.0,
    "left_forearm_roll": 22.0,
    "left_wrist_angle": 35.0,
    "left_wrist_rotate": 35.0,
    "left_left_finger": 35.0,
    "left_right_finger": 35.0,
}

# Default joint positions (rest pose)
_DEFAULT_JOINT_POS = {
    "left_waist": 0.0,
    "left_shoulder": -0.96,
    "left_elbow": 1.16,
    "left_forearm_roll": 0.0,
    "left_wrist_angle": -0.3,
    "left_wrist_rotate": 0.0,
    "left_left_finger": 0.0084,
    "left_right_finger": 0.0084,
}


def _build_pickup_robot_cfg() -> ArticulationCfg:
    """Build ArticulationCfg for the pickup scene USD.

    The scene USD contains the full environment (robot + table + rack + tube).
    The articulation root is the table xform, which includes the robot as a
    fixed child with 8 physics joints.
    """
    # Per-joint actuator configs (Isaac Lab requires float or dict, not list)
    _ARM_JOINT_NAMES = _JOINT_NAMES[:6]

    actuators = {}
    for jname in _ARM_JOINT_NAMES:
        actuators[jname] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=_STIFFNESS[jname],
            damping=_DAMPING[jname],
            effort_limit=_EFFORT[jname],
        )
    actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=["left_left_finger", "left_right_finger"],
        stiffness=2000.0,
        damping=124.0,
        effort_limit=35.0,
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PICKUP_SCENE_USD,
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


def _build_tube_body_cfg(prim_path: str) -> RigidObjectCfg:
    """Build RigidObjectCfg for the tube body rigid body in the scene."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_PICKUP_SCENE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
    )


# Default robot configuration for the pickup scene
PICKUP_ROBOT_CFG = _build_pickup_robot_cfg()


@configclass
class PickupEnvCfg(AutobioEnvCfg):
    """Configuration for the pickup task."""

    task_name = "pickup"
    episode_length_s = 22.5
    early_stop = True

    # Observation: 6 arm pos + 6 arm vel + 2 gripper pos + 3 tube body pos = 17
    observation_space = 17
    action_space = 8  # 6 arm DOF + 2 gripper

    # Success threshold: tube body z-position
    tube_height_threshold: float = 0.925

    # Randomization ranges for arm joint perturbation
    arm_perturbation_low: tuple[float, ...] = (-0.2, 0.0, 0.0, -0.05, 0.0, -0.1)
    arm_perturbation_high: tuple[float, ...] = (0.2, 0.3, 0.1, 0.05, 0.3, 0.1)

    # Rack grid parameters
    rack_rows: int = 2
    rack_cols: int = 5

    # Robot: load the pickup scene USD (contains robot + table + rack + tube)
    robot_cfg: ArticulationCfg = PICKUP_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class PickupEnv(AutobioEnv):
    """Pickup task environment.

    The pickup scene USD embeds the robot, table, rack, and tube as a single
    articulation.  Joint indices 0-5 are the arm DOFs and 6-7 are the gripper
    fingers.  The tube body is tracked separately via RigidObject.
    """

    cfg: PickupEnvCfg

    def __init__(self, cfg: PickupEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Arm DOF indices: joints 0-5
        self._arm_dof_indices = list(range(6))

        # Gripper DOF indices: joints 6-7
        self._gripper_dof_indices = [6, 7]

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
        # Tube body is tracked via robot body poses, not as separate RigidObject

    def _get_observations(self) -> dict:
        """Return observations: joint pos + joint vel + tube body position."""
        arm_pos = self.joint_pos[:, self._arm_dof_indices]
        arm_vel = self.joint_vel[:, self._arm_dof_indices]
        gripper_pos = self.joint_pos[:, self._gripper_dof_indices]

        # Tube body position via articulation body tracking
        try:
            tube_pos, _ = self.get_body_pose("tn__2_centrifuge_50ml_screw_body_")
        except Exception:
            tube_pos = torch.zeros(self.num_envs, 3, device=self.device)

        obs = torch.cat([arm_pos, arm_vel, gripper_pos, tube_pos[:, :3]], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Sparse reward: 1.0 if tube body z-position exceeds threshold."""
        try:
            tube_pos, _ = self.get_body_pose("tn__2_centrifuge_50ml_screw_body_")
            tube_z = tube_pos[:, 2]
        except Exception:
            tube_z = torch.zeros(self.num_envs, device=self.device)
        success = tube_z > self.cfg.tube_height_threshold
        return success.float()

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
