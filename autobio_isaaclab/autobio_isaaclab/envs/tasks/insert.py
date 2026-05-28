"""Insert task: transfer a centrifuge tube from one rack to another.

Ports AutoBio's ``Insert`` task to Isaac Lab DirectRLEnv.

Original task: ``AutoBio/autobio/insert_centrifuge_tube.py``
Scene: ``usd_assets/insert/insert/insert.usda``

The agent controls a single UR5e arm (6 DOF + Robotiq 2F-85 gripper) to
pick up a 50ml centrifuge tube from one 10-slot rack and place it into the
other rack.

Scene USD contains the full environment (robot + table + 2 racks + tube)
as a single articulation.  The tube body and cap are separate rigid bodies
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

# Path to the insert scene USD
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_INSERT_SCENE_USD = os.path.join(_USD_DIR, "insert", "insert", "insert.usda")

# UR5e arm joint names (6-DOF, matching MJCF naming)
_UR5E_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Robotiq 2F-85 gripper joint names (all linkage joints in the USD)
_UR5E_GRIPPER_JOINT_NAMES = [
    "right_driver_joint",
    "left_driver_joint",
    "right_coupler_joint",
    "left_coupler_joint",
    "right_spring_link_joint",
    "left_spring_link_joint",
    "right_follower_joint",
    "left_follower_joint",
]

# All controllable joints
_UR5E_JOINT_NAMES = _UR5E_ARM_JOINT_NAMES + _UR5E_GRIPPER_JOINT_NAMES

# Per-joint stiffness (kp) for arm
_ARM_STIFFNESS = {
    "shoulder_pan": 100.0,
    "shoulder_lift": 100.0,
    "elbow": 100.0,
    "wrist_1": 100.0,
    "wrist_2": 100.0,
    "wrist_3": 100.0,
}

# Per-joint damping (kd) for arm
_ARM_DAMPING = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 10.0,
    "elbow": 10.0,
    "wrist_1": 10.0,
    "wrist_2": 10.0,
    "wrist_3": 10.0,
}

# Effort limits for arm (Nm)
_ARM_EFFORT = {
    "shoulder_pan": 150.0,
    "shoulder_lift": 150.0,
    "elbow": 150.0,
    "wrist_1": 28.0,
    "wrist_2": 28.0,
    "wrist_3": 28.0,
}

# Default joint positions (home pose -- arm slightly raised, gripper open)
_DEFAULT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.5708,  # -90 deg
    "elbow": 1.5708,           # 90 deg
    "wrist_1": -1.5708,        # -90 deg
    "wrist_2": -1.5708,        # -90 deg
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


def _build_insert_robot_cfg() -> ArticulationCfg:
    """Build ArticulationCfg for the insert scene USD.

    The scene USD contains the full environment (robot + table + 2 racks + tube).
    The articulation root is the table xform, which includes the UR5e as a
    fixed child with 14 physics joints (6 arm + 8 Robotiq 2F-85 gripper).
    """
    actuators = {}
    for jname in _UR5E_ARM_JOINT_NAMES:
        actuators[jname] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=_ARM_STIFFNESS[jname],
            damping=_ARM_DAMPING[jname],
            effort_limit_sim=_ARM_EFFORT[jname],
        )
    # All gripper joints (driver + linkage followers)
    actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=_UR5E_GRIPPER_JOINT_NAMES,
        stiffness=2000.0,
        damping=100.0,
        effort_limit_sim=20.0,
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_INSERT_SCENE_USD,
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
            usd_path=_INSERT_SCENE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
    )


# Default robot configuration for the insert scene
INSERT_ROBOT_CFG = _build_insert_robot_cfg()


@configclass
class InsertEnvCfg(AutobioEnvCfg):
    """Configuration for the insert (transfer centrifuge tube) task."""

    task_name = "insert"
    episode_length_s = 15.0
    early_stop = True

    # Observation: 6 arm joint pos + 6 arm joint vel + 8 gripper pos + 3 tube body pos = 23
    observation_space = 23
    action_space = 14  # 6 arm DOF + 8 gripper joints

    # Success thresholds
    tube_z_min: float = 0.829
    tube_z_max: float = 0.831
    tube_xy_threshold: float = 0.015  # max XY distance to target rack center

    # Target rack center position (world XY, second rack)
    target_rack_pos: tuple[float, float] = (0.55, 0.0)

    # Randomization ranges for arm joint perturbation
    arm_perturbation_low: tuple[float, ...] = (-0.2, -0.15, -0.1, -0.05, -0.05, -0.1)
    arm_perturbation_high: tuple[float, ...] = (0.2, 0.15, 0.1, 0.05, 0.05, 0.1)

    # Robot: load the insert scene USD (contains UR5e + table + 2 racks + tube)
    robot_cfg: ArticulationCfg = INSERT_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class InsertEnv(AutobioEnv):
    """Insert task environment.

    The insert scene USD embeds the UR5e robot, table, two racks, and tube
    as a single articulation.  Joint indices 0-5 are the arm DOFs and 6-7
    are the Robotiq 2F-85 gripper fingers.  The tube body is tracked
    separately via RigidObject.
    """

    cfg: InsertEnvCfg

    def __init__(self, cfg: InsertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Arm DOF indices: joints 0-5
        self._arm_dof_indices = list(range(6))

        # Gripper DOF indices: joints 6-13 (8 Robotiq 2F-85 linkage joints)
        self._gripper_dof_indices = list(range(6, 14))

        # Target rack center (XY) as tensor
        self._target_rack_xy = torch.tensor(
            self.cfg.target_rack_pos, device=self.device, dtype=torch.float32
        )

        # Perturbation tensors
        self._perturb_low = torch.tensor(
            self.cfg.arm_perturbation_low, device=self.device, dtype=torch.float32
        )
        self._perturb_high = torch.tensor(
            self.cfg.arm_perturbation_high, device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        """Set up the insert scene."""
        super()._setup_scene()

        # Tube body: separate rigid body in the scene USD
        tube_body_prim = self.cfg.robot_cfg.prim_path.replace(
            "/Robot", "/Robot/Geometry/centrifuge_50ml_screw_body_wrap/tn__2_centrifuge_50ml_screw_body_"
        )
        self.tube_body = RigidObject(
            _build_tube_body_cfg(tube_body_prim)
        )
        self.scene.rigid_objects["tube_body"] = self.tube_body

    def _get_observations(self) -> dict:
        """Return observations: joint pos + joint vel + tube body position."""
        arm_pos = self.joint_pos[:, self._arm_dof_indices]
        arm_vel = self.joint_vel[:, self._arm_dof_indices]
        gripper_pos = self.joint_pos[:, self._gripper_dof_indices]

        # Tube body position relative to env origin
        tube_pos = self.tube_body.data.root_pos_w[:, :3] - self.scene.env_origins

        obs = torch.cat([arm_pos, arm_vel, gripper_pos, tube_pos], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Sparse reward: 1.0 if tube is placed in target rack.

        Success requires:
          - tube body z in (tube_z_min, tube_z_max)
          - tube body XY distance to target rack center < tube_xy_threshold
        """
        tube_pos = self.tube_body.data.root_pos_w[:, :3] - self.scene.env_origins
        tube_z = tube_pos[:, 2]
        tube_xy = tube_pos[:, :2]

        # Z height check
        z_ok = (tube_z > self.cfg.tube_z_min) & (tube_z < self.cfg.tube_z_max)

        # XY distance to target rack center
        xy_dist = torch.norm(tube_xy - self._target_rack_xy.unsqueeze(0), dim=-1)
        xy_ok = xy_dist < self.cfg.tube_xy_threshold

        success = z_ok & xy_ok
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
