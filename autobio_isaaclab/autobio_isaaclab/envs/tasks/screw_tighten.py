"""Screw-tighten task: tighten a centrifuge tube cap with dual ALOHA arms.

Ports AutoBio's ``ScrewTighten`` task to Isaac Lab DirectRLEnv.

Scene: ``usd_assets/lab_screw_tighten/lab_screw_tighten/lab_screw_tighten.usda``

The agent controls two ALOHA arms (14 DOF total: 7 per arm) to screw the
cap onto a 50ml centrifuge tube.  Arm1 grips and rotates the cap while
arm2 holds the tube body steady.

Success criteria:
  - Cap-body z-height < 0.123
  - Contact between nut and bolt (screw engaged)
  - XY distance cap to tube < 0.005
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import sample_uniform

from ..autobio_dual_env import ALL_DOF_INDICES, AutobioDualEnv
from ..autobio_dual_env_cfg import AutobioDualEnvCfg

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Scene USD path
# ---------------------------------------------------------------------------
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_SCREW_TIGHTEN_SCENE_USD = os.path.join(_USD_DIR, "lab_screw_tighten", "lab_screw_tighten", "lab_screw_tighten.usda")

# ---------------------------------------------------------------------------
# Joint names per arm (order matches articulation indices)
# ---------------------------------------------------------------------------
ARM1_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_left_finger",
    "left_right_finger",
]

ARM2_JOINT_NAMES = [
    "left_waist_2",
    "left_shoulder_2",
    "left_elbow_2",
    "left_forearm_roll_2",
    "left_wrist_angle_2",
    "left_wrist_rotate_2",
    "left_left_finger_2",
    "left_right_finger_2",
]

# ---------------------------------------------------------------------------
# Actuator parameters per arm joint (kp, kd, effort)
# ---------------------------------------------------------------------------
_ARM_STIFFNESS = [43.0, 265.0, 227.0, 78.0, 37.0, 10.4]
_ARM_DAMPING = [5.76, 20.0, 18.49, 6.78, 6.28, 1.2]
_ARM_EFFORT = [35.0, 144.0, 59.0, 22.0, 35.0, 35.0]
_GRIPPER_STIFFNESS = 2000.0
_GRIPPER_DAMPING = 124.0
_GRIPPER_EFFORT = 35.0

# ---------------------------------------------------------------------------
# Default joint positions (rest pose, both arms)
# ---------------------------------------------------------------------------
_DEFAULT_JOINT_POS = {
    # Arm 1
    "left_waist": 0.0,
    "left_shoulder": -0.96,
    "left_elbow": 1.16,
    "left_forearm_roll": 0.0,
    "left_wrist_angle": -0.3,
    "left_wrist_rotate": 0.0,
    "left_left_finger": 0.0084,
    "left_right_finger": 0.0084,
    # Arm 2
    "left_waist_2": 0.0,
    "left_shoulder_2": -0.96,
    "left_elbow_2": 1.16,
    "left_forearm_roll_2": 0.0,
    "left_wrist_angle_2": -0.3,
    "left_wrist_rotate_2": 0.0,
    "left_left_finger_2": 0.0084,
    "left_right_finger_2": 0.0084,
}


def _build_screw_tighten_actuators() -> dict:
    """Build per-joint actuator configs for the dual-arm screw-tighten scene."""
    actuators = {}
    for i, jname in enumerate(ARM1_JOINT_NAMES[:6]):
        actuators[f"arm1_{jname}"] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=_ARM_STIFFNESS[i],
            damping=_ARM_DAMPING[i],
            effort_limit=_ARM_EFFORT[i],
        )
    actuators["gripper1"] = ImplicitActuatorCfg(
        joint_names_expr=["left_left_finger", "left_right_finger"],
        stiffness=_GRIPPER_STIFFNESS,
        damping=_GRIPPER_DAMPING,
        effort_limit=_GRIPPER_EFFORT,
    )
    for i, jname in enumerate(ARM2_JOINT_NAMES[:6]):
        actuators[f"arm2_{jname}"] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=_ARM_STIFFNESS[i],
            damping=_ARM_DAMPING[i],
            effort_limit=_ARM_EFFORT[i],
        )
    actuators["gripper2"] = ImplicitActuatorCfg(
        joint_names_expr=["left_left_finger_2", "left_right_finger_2"],
        stiffness=_GRIPPER_STIFFNESS,
        damping=_GRIPPER_DAMPING,
        effort_limit=_GRIPPER_EFFORT,
    )
    return actuators


def _build_screw_tighten_robot_cfg() -> ArticulationCfg:
    """Build ArticulationCfg for the screw-tighten scene USD."""
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_SCREW_TIGHTEN_SCENE_USD,
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
        actuators=_build_screw_tighten_actuators(),
    )


def _build_rigid_object_cfg(prim_path: str) -> RigidObjectCfg:
    """Build RigidObjectCfg for a rigid body in the screw-tighten scene."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=_SCREW_TIGHTEN_SCENE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
    )


# Default robot configuration for the screw-tighten scene
SCREW_TIGHTEN_ROBOT_CFG = _build_screw_tighten_robot_cfg()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@configclass
class ScrewTightenEnvCfg(AutobioDualEnvCfg):
    """Configuration for the screw-tighten task."""

    task_name = "screw_tighten"
    episode_length_s = 45.0
    early_stop = True

    # Observation: 14 joint pos + 14 joint vel + 3 cap pos + 3 body pos = 34
    observation_space = 34
    action_space = 14  # 7 per arm x 2 arms

    # Success thresholds
    cap_body_z_threshold: float = 0.123  # cap must be within this z-distance
    cap_body_xy_threshold: float = 0.005  # cap XY must align with body
    nut_bolt_z_threshold: float = 0.01  # nut-bolt z-distance for engagement

    # Randomization ranges for arm joint perturbation
    arm1_perturbation_low: tuple[float, ...] = (-0.2, 0.0, 0.0, -0.05, 0.0, -0.1)
    arm1_perturbation_high: tuple[float, ...] = (0.2, 0.3, 0.1, 0.05, 0.3, 0.1)
    arm2_perturbation_low: tuple[float, ...] = (-0.2, 0.0, 0.0, -0.05, 0.0, -0.1)
    arm2_perturbation_high: tuple[float, ...] = (0.2, 0.3, 0.1, 0.05, 0.3, 0.1)

    # Robot: load the screw-tighten scene USD (dual-arm robot + table + tube)
    robot_cfg: ArticulationCfg = SCREW_TIGHTEN_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ScrewTightenEnv(AutobioDualEnv):
    """Screw-tighten task environment.

    The screw-tighten scene USD embeds the dual-arm robot, table, and centrifuge
    tube (cap + body) as a single articulation.  Joint indices 0-5 are arm1,
    6-7 are arm1 gripper, 8-13 are arm2, 14-15 are arm2 gripper.  The cap
    and body are tracked separately via RigidObject.
    """

    cfg: ScrewTightenEnvCfg

    def __init__(self, cfg: ScrewTightenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Perturbation tensors
        self._arm1_perturb_low = torch.tensor(
            self.cfg.arm1_perturbation_low, device=self.device, dtype=torch.float32
        )
        self._arm1_perturb_high = torch.tensor(
            self.cfg.arm1_perturbation_high, device=self.device, dtype=torch.float32
        )
        self._arm2_perturb_low = torch.tensor(
            self.cfg.arm2_perturbation_low, device=self.device, dtype=torch.float32
        )
        self._arm2_perturb_high = torch.tensor(
            self.cfg.arm2_perturbation_high, device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        """Set up the screw-tighten scene."""
        super()._setup_scene()

        # Cap: separate rigid body in the scene USD
        cap_prim = self.cfg.robot_cfg.prim_path.replace(
            "/Robot", "/Robot/Geometry/centrifuge_50ml_screw_cap_wrap/tn__6_centrifuge_50ml_screw_cap_"
        )
        self.cap = RigidObject(_build_rigid_object_cfg(cap_prim))
        self.scene.rigid_objects["cap"] = self.cap

        # Tube body: separate rigid body in the scene USD
        body_prim = self.cfg.robot_cfg.prim_path.replace(
            "/Robot", "/Robot/Geometry/centrifuge_50ml_screw_body_wrap/tn__7_centrifuge_50ml_screw_body_"
        )
        self.tube_body = RigidObject(_build_rigid_object_cfg(body_prim))
        self.scene.rigid_objects["tube_body"] = self.tube_body

    def _get_observations(self) -> dict:
        """Return observations: joint pos + joint vel + cap pos + body pos."""
        # Joint positions and velocities (14 each)
        joint_pos = self.joint_pos[:, ALL_DOF_INDICES]
        joint_vel = self.joint_vel[:, ALL_DOF_INDICES]

        # Cap and body positions relative to env origin
        cap_pos = self.cap.data.root_pos_w[:, :3] - self.scene.env_origins
        body_pos = self.tube_body.data.root_pos_w[:, :3] - self.scene.env_origins

        obs = torch.cat([joint_pos, joint_vel, cap_pos, body_pos], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Sparse reward: 1.0 when cap is fully tightened.

        Success requires all three conditions:
          1. Cap-body z-height difference < threshold (cap seated)
          2. Nut-bolt z-distance < threshold (screw threads engaged)
          3. Cap XY distance to body < threshold (cap aligned)
        """
        cap_pos = self.cap.data.root_pos_w[:, :3]
        body_pos = self.tube_body.data.root_pos_w[:, :3]

        # Condition 1: cap close to body in z (screwed down)
        cap_body_z = torch.abs(cap_pos[:, 2] - body_pos[:, 2])
        z_ok = cap_body_z < self.cfg.cap_body_z_threshold

        # Condition 2: nut and bolt close in z (threads engaged)
        # Nut is part of cap wrap, bolt is part of body wrap.
        # Use root positions as proxy for nut/bolt positions.
        nut_bolt_z = torch.abs(cap_pos[:, 2] - body_pos[:, 2])
        engaged = nut_bolt_z < self.cfg.nut_bolt_z_threshold

        # Condition 3: cap aligned with body in XY
        cap_body_xy = torch.norm(cap_pos[:, :2] - body_pos[:, :2], dim=-1)
        xy_ok = cap_body_xy < self.cfg.cap_body_xy_threshold

        success = z_ok & engaged & xy_ok
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

        arm1_perturbation = sample_uniform(
            self._arm1_perturb_low,
            self._arm1_perturb_high,
            (num_reset, 6),
            self.device,
        )
        arm2_perturbation = sample_uniform(
            self._arm2_perturb_low,
            self._arm2_perturb_high,
            (num_reset, 6),
            self.device,
        )

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()

        # Apply perturbation to arm1 joints (indices 0-5)
        joint_pos[:, :6] += arm1_perturbation
        # Apply perturbation to arm2 joints (indices 8-13, skip gripper at 6-7)
        joint_pos[:, 8:14] += arm2_perturbation

        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
