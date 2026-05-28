"""Thermal cycler manipulation task: open or close a Bio-Rad C1000 lid.

Ports AutoBio's ``ThermalCyclerManipulate`` task to Isaac Lab DirectRLEnv.

Original task: ``AutoBio/autobio/mani_thermal_cycler.py``
Scene: ``usd_assets/mani_thermal_cycler/mani_thermal_cycler/mani_thermal_cycler.usda``

The agent controls a single UR5e arm (6 DOF + 2F-85 gripper) to open or close
the lid of a Bio-Rad C1000 thermal cycler.  The scene contains the full
environment (robot + table + thermal cycler) as a single articulation.

Two task variants:
- ``thermal_cycler_close``: close the lid (time limit 31.5s)
- ``thermal_cycler_open``: open the lid (time limit 22.5s)
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

# Path to the thermal cycler scene USD
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_THERMAL_CYCLER_SCENE_USD = os.path.join(
    _USD_DIR, "mani_thermal_cycler", "mani_thermal_cycler", "mani_thermal_cycler.usda"
)

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

# All controllable joint names
_JOINT_NAMES = _UR5E_ARM_JOINT_NAMES + _UR5E_GRIPPER_JOINT_NAMES

# Per-joint stiffness (kp) from MuJoCo defaults
_STIFFNESS = {
    "shoulder_pan": 100.0,
    "shoulder_lift": 100.0,
    "elbow": 100.0,
    "wrist_1": 100.0,
    "wrist_2": 100.0,
    "wrist_3": 100.0,
    "right_driver_joint": 2000.0,
    "left_driver_joint": 2000.0,
    "right_coupler_joint": 2000.0,
    "left_coupler_joint": 2000.0,
    "right_spring_link_joint": 2000.0,
    "left_spring_link_joint": 2000.0,
    "right_follower_joint": 2000.0,
    "left_follower_joint": 2000.0,
}

# Per-joint damping (kd) from MuJoCo defaults
_DAMPING = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 10.0,
    "elbow": 10.0,
    "wrist_1": 10.0,
    "wrist_2": 10.0,
    "wrist_3": 10.0,
    "right_driver_joint": 100.0,
    "left_driver_joint": 100.0,
    "right_coupler_joint": 100.0,
    "left_coupler_joint": 100.0,
    "right_spring_link_joint": 100.0,
    "left_spring_link_joint": 100.0,
    "right_follower_joint": 100.0,
    "left_follower_joint": 100.0,
}

# Effort limits from UR5e datasheet (Nm)
_EFFORT = {
    "shoulder_pan": 150.0,
    "shoulder_lift": 150.0,
    "elbow": 150.0,
    "wrist_1": 28.0,
    "wrist_2": 28.0,
    "wrist_3": 28.0,
    "right_driver_joint": 20.0,
    "left_driver_joint": 20.0,
    "right_coupler_joint": 20.0,
    "left_coupler_joint": 20.0,
    "right_spring_link_joint": 20.0,
    "left_spring_link_joint": 20.0,
    "right_follower_joint": 20.0,
    "left_follower_joint": 20.0,
}

# Default initial joint positions (home pose -- arm slightly raised)
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

# Instrument joint names in the scene USD
# These are the physics joint names for the thermal cycler lid mechanism
_INSTRUMENT_LID_JOINT = "lid"
_INSTRUMENT_LEVER_JOINT = "lid_lever_1"


def _build_thermal_cycler_robot_cfg() -> ArticulationCfg:
    """Build ArticulationCfg for the thermal cycler scene USD.

    The scene USD contains the full environment (robot + table + thermal cycler).
    The articulation root is the table xform, which includes the robot as a
    fixed child with 8 physics joints.
    """
    actuator_cfgs = []

    # Arm joints -- per-joint stiffness/damping
    for jname in _UR5E_ARM_JOINT_NAMES:
        actuator_cfgs.append(
            ImplicitActuatorCfg(
                joint_names_expr=[jname],
                stiffness=_STIFFNESS[jname],
                damping=_DAMPING[jname],
                effort_limit=_EFFORT[jname],
            )
        )

    # Gripper joints -- uniform high stiffness
    actuator_cfgs.append(
        ImplicitActuatorCfg(
            joint_names_expr=_UR5E_GRIPPER_JOINT_NAMES,
            stiffness=2000.0,
            damping=100.0,
            effort_limit=20.0,
        )
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_THERMAL_CYCLER_SCENE_USD,
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
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=_UR5E_ARM_JOINT_NAMES,
                stiffness=100.0,
                damping=10.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=_UR5E_GRIPPER_JOINT_NAMES,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )


# Default robot configuration for the thermal cycler scene
THERMAL_CYCLER_ROBOT_CFG = _build_thermal_cycler_robot_cfg()


@configclass
class ThermalCyclerEnvCfg(AutobioEnvCfg):
    """Configuration for the thermal cycler manipulation task."""

    task_name = "thermal_cycler"
    early_stop = True

    # Task variant: "thermal_cycler_close" or "thermal_cycler_open"
    task_variant: str = "thermal_cycler_close"

    # Observation: 6 arm joint pos + 6 arm joint vel + 2 gripper pos + 2 instrument joint pos = 16
    observation_space = 21
    action_space = 14  # 6 arm DOF + 8 gripper linkage joints

    # Success thresholds for close variant
    # abs(lid_qpos + 1.9) < 0.01 AND abs(lever_qpos + 0.94) < 0.01
    close_lid_target: float = -1.9
    close_lever_target: float = -0.94
    close_lid_threshold: float = 0.01
    close_lever_threshold: float = 0.01

    # Success thresholds for open variant
    # abs(lid_qpos) < 0.003 AND abs(lever_qpos) < 0.003
    open_lid_target: float = 0.0
    open_lever_target: float = 0.0
    open_lid_threshold: float = 0.003
    open_lever_threshold: float = 0.003

    # Time limits per variant
    close_time_limit: float = 31.5
    open_time_limit: float = 22.5

    # Randomization ranges for arm joint perturbation
    # From original MuJoCo task: lows=(-1.2,-0.2,-0.1,-0.5,-0.2,-0.2), highs=(0.0,0.0,0.1,0.2,0.2,0.2)
    arm_perturbation_low: tuple[float, ...] = (-1.2, -0.2, -0.1, -0.5, -0.2, -0.2)
    arm_perturbation_high: tuple[float, ...] = (0.0, 0.0, 0.1, 0.2, 0.2, 0.2)

    # Lid/lever initial positions for open task variant
    # When task is "open", lid starts at closed position (lid_jntlimit[0] = -1.9)
    open_lid_initial: float = -1.9
    open_lever_initial: float = -0.94

    # Robot: load the thermal cycler scene USD (contains robot + table + thermal cycler)
    robot_cfg: ArticulationCfg = THERMAL_CYCLER_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class ThermalCyclerEnv(AutobioEnv):
    """Thermal cycler manipulation task environment.

    The thermal cycler scene USD embeds the robot, table, and thermal cycler
    as a single articulation.  Joint indices 0-5 are the arm DOFs and 6-7 are
    the gripper fingers.  The instrument joints (lid, lever) are passive joints
    in the same articulation.

    Task variants:
    - ``thermal_cycler_close``: close the lid from open position
    - ``thermal_cycler_open``: open the lid from closed position
    """

    cfg: ThermalCyclerEnvCfg

    def __init__(self, cfg: ThermalCyclerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Arm DOF indices: joints 0-5
        self._arm_dof_indices = list(range(6))

        # Gripper DOF indices: joints 6-7
        self._gripper_dof_indices = list(range(6, 14))

        # Perturbation tensors
        self._perturb_low = torch.tensor(
            self.cfg.arm_perturbation_low, device=self.device, dtype=torch.float32
        )
        self._perturb_high = torch.tensor(
            self.cfg.arm_perturbation_high, device=self.device, dtype=torch.float32
        )

        # Find instrument joint indices in the articulation
        # These are passive joints in the same articulation as the robot
        self._lid_joint_idx = self._find_joint_index(_INSTRUMENT_LID_JOINT)
        self._lever_joint_idx = self._find_joint_index(_INSTRUMENT_LEVER_JOINT)

        # Set time limit based on task variant
        if self.cfg.task_variant == "thermal_cycler_close":
            self.cfg.episode_length_s = self.cfg.close_time_limit
        elif self.cfg.task_variant == "thermal_cycler_open":
            self.cfg.episode_length_s = self.cfg.open_time_limit
        else:
            raise ValueError(f"Unknown task variant: {self.cfg.task_variant}")

    def _find_joint_index(self, joint_name: str) -> int:
        """Find the index of a joint by name in the articulation.

        Args:
            joint_name: Name of the joint to find.

        Returns:
            Index of the joint in the joint tensors.

        Raises:
            ValueError: If joint not found.
        """
        joint_names = self.robot.data.joint_names
        for i, name in enumerate(joint_names):
            if name == joint_name:
                return i
        raise ValueError(
            f"Joint '{joint_name}' not found in articulation. "
            f"Available joints: {joint_names}"
        )

    def _get_observations(self) -> dict:
        """Return observations: arm joint pos + arm joint vel + gripper pos + instrument joint pos."""
        arm_pos = self.joint_pos[:, self._arm_dof_indices]
        arm_vel = self.joint_vel[:, self._arm_dof_indices]
        gripper_pos = self.joint_pos[:, self._gripper_dof_indices]

        # Instrument joint positions (lid and lever)
        lid_qpos = self.joint_pos[:, self._lid_joint_idx].unsqueeze(1)
        lever_qpos = self.joint_pos[:, self._lever_joint_idx].unsqueeze(1)

        obs = torch.cat([arm_pos, arm_vel, gripper_pos, lid_qpos, lever_qpos], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Sparse reward: 1.0 if success condition is met."""
        lid_qpos = self.joint_pos[:, self._lid_joint_idx]
        lever_qpos = self.joint_pos[:, self._lever_joint_idx]

        if self.cfg.task_variant == "thermal_cycler_close":
            lid_ok = torch.abs(lid_qpos - self.cfg.close_lid_target) < self.cfg.close_lid_threshold
            lever_ok = torch.abs(lever_qpos - self.cfg.close_lever_target) < self.cfg.close_lever_threshold
        elif self.cfg.task_variant == "thermal_cycler_open":
            lid_ok = torch.abs(lid_qpos - self.cfg.open_lid_target) < self.cfg.open_lid_threshold
            lever_ok = torch.abs(lever_qpos - self.cfg.open_lever_target) < self.cfg.open_lever_threshold
        else:
            raise ValueError(f"Unknown task variant: {self.cfg.task_variant}")

        success = lid_ok & lever_ok
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

        num_reset = len(env_ids)

        # Randomize arm joint positions with perturbation
        perturbation = sample_uniform(
            self._perturb_low,
            self._perturb_high,
            (num_reset, len(self._arm_dof_indices)),
            self.device,
        )

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._arm_dof_indices] += perturbation

        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        # Set instrument joint positions based on task variant
        if self.cfg.task_variant == "thermal_cycler_open":
            # For open task: lid starts at closed position
            joint_pos[:, self._lid_joint_idx] = self.cfg.open_lid_initial
            joint_pos[:, self._lever_joint_idx] = self.cfg.open_lever_initial

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
