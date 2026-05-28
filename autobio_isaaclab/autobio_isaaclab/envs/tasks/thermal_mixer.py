"""Thermal mixer task: press buttons on an Eppendorf Thermomixer C.

Ports AutoBio's thermal mixer task to Isaac Lab DirectRLEnv.

The agent controls a single UR5e arm (6 DOF) with a Robotiq 2F-85 gripper
to press buttons on a Thermomixer C, setting temperature, time, and
shaking frequency to target values.

Scene: ``usd_assets/mani_thermal_mixer/mani_thermal_mixer/mani_thermal_mixer.usda``

Buttons (6 total):
  speed_up, speed_down, temp_up, temp_down, time_up, time_down

Reward: continuous [0, 1] weighted composite:
  - Temperature: weight 0.5, score = max(0, 1 - |current - target| / |target - 25|)
  - Time:        weight 0.3, score = max(0, 1 - |current - target| / |target - 60|)
  - Frequency:   weight 0.2, score = max(0, 1 - |current - target| / |target - 900|)

NOTE: The full reward requires reading the mixer's display state from the
simulation, which depends on a UI state machine not yet implemented. The
reward is currently a placeholder returning 0.0.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import sample_uniform

from ..autobio_env import AutobioEnv
from ..autobio_env_cfg import AutobioEnvCfg
from ...assets.ur5e import (
    UR5E_ARM_JOINT_NAMES,
    UR5E_GRIPPER_JOINT_NAMES,
    UR5E_JOINT_NAMES,
    make_ur5e_2f85_cfg,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Scene USD path
# ---------------------------------------------------------------------------

_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "usd_assets")
_THERMAL_MIXER_SCENE_USD = os.path.join(
    _USD_DIR, "mani_thermal_mixer", "mani_thermal_mixer", "mani_thermal_mixer.usda"
)

# ---------------------------------------------------------------------------
# Joint structure: 6 arm + 2 gripper = 8 DOF
# ---------------------------------------------------------------------------

_ARM_DOF_COUNT = len(UR5E_ARM_JOINT_NAMES)   # 6
_GRIPPER_DOF_COUNT = len(UR5E_GRIPPER_JOINT_NAMES)  # 2
_TOTAL_DOF = _ARM_DOF_COUNT + _GRIPPER_DOF_COUNT  # 8

# Action space: 6 arm position targets + 8 gripper joint targets
_ACTION_SPACE = _ARM_DOF_COUNT + len(UR5E_GRIPPER_JOINT_NAMES)  # 14

# Observation space: 6 arm pos + 6 arm vel + 2 gripper pos + 3 ee pos = 17
_OBSERVATION_SPACE = _ARM_DOF_COUNT + _ARM_DOF_COUNT + _GRIPPER_DOF_COUNT + 3  # 17

# ---------------------------------------------------------------------------
# Default joint positions (home pose for thermal mixer task)
# ---------------------------------------------------------------------------

_DEFAULT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.5708,
    "elbow": 1.5708,
    "wrist_1": -1.5708,
    "wrist_2": -1.5708,
    "wrist_3": 0.0,
    "right_driver_joint": 0.0,
    "left_driver_joint": 0.0,
}

# ---------------------------------------------------------------------------
# Target parameter ranges (for randomization on reset)
# ---------------------------------------------------------------------------

# Temperature: 25-95 C (Thermomixer C range)
TARGET_TEMP_LOW = 30.0
TARGET_TEMP_HIGH = 90.0

# Time: 60-999 s
TARGET_TIME_LOW = 60.0
TARGET_TIME_HIGH = 600.0

# Frequency/shaking speed: 300-1500 rpm (default 900)
TARGET_FREQ_LOW = 300.0
TARGET_FREQ_HIGH = 1500.0

# Mixer display defaults (cold start)
DEFAULT_TEMP = 25.0
DEFAULT_TIME = 60.0
DEFAULT_FREQ = 900.0


def _build_thermal_mixer_robot_cfg() -> ArticulationCfg:
    """Build ArticulationCfg for the thermal mixer scene.

    Uses the UR5e + 2F-85 asset configuration, pointing at the thermal
    mixer scene USD.
    """
    return make_ur5e_2f85_cfg(
        usd_path=_THERMAL_MIXER_SCENE_USD,
        prim_path="/World/envs/env_.*/Robot",
    )


# Default robot configuration for the thermal mixer scene
THERMAL_MIXER_ROBOT_CFG = _build_thermal_mixer_robot_cfg()


@configclass
class ThermalMixerEnvCfg(AutobioEnvCfg):
    """Configuration for the thermal mixer task."""

    task_name = "thermal_mixer"
    episode_length_s = 30.0
    early_stop = False

    observation_space = _OBSERVATION_SPACE
    action_space = _ACTION_SPACE

    # Reward component weights
    reward_weight_temp: float = 0.5
    reward_weight_time: float = 0.3
    reward_weight_freq: float = 0.2

    # Randomization ranges for arm joint perturbation (UR5e home pose)
    arm_perturbation_low: tuple[float, ...] = (-0.15, -0.1, -0.1, -0.1, -0.1, -0.1)
    arm_perturbation_high: tuple[float, ...] = (0.15, 0.1, 0.1, 0.1, 0.1, 0.1)

    # Robot: load the thermal mixer scene USD
    robot_cfg: ArticulationCfg = THERMAL_MIXER_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class ThermalMixerEnv(AutobioEnv):
    """Thermal mixer task environment.

    The scene USD contains the UR5e robot, table, and Thermomixer C as a
    single articulation.  Joint indices 0-5 are the arm DOFs and 6-7 are
    the gripper fingers.
    """

    cfg: ThermalMixerEnvCfg

    def __init__(self, cfg: ThermalMixerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Arm DOF indices: joints 0-5
        self._arm_dof_indices = list(range(_ARM_DOF_COUNT))

        # Gripper DOF indices: joints 6-7
        self._gripper_dof_indices = list(
            range(_ARM_DOF_COUNT, _TOTAL_DOF)
        )

        # Perturbation tensors
        self._perturb_low = torch.tensor(
            self.cfg.arm_perturbation_low, device=self.device, dtype=torch.float32
        )
        self._perturb_high = torch.tensor(
            self.cfg.arm_perturbation_high, device=self.device, dtype=torch.float32
        )

        # Target parameters (randomized on reset)
        # Shape: (num_envs, 3) -- [target_temp, target_time, target_freq]
        self._target_params = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )

        # Current mixer display state (placeholder -- not yet read from sim)
        # Shape: (num_envs, 3) -- [current_temp, current_time, current_freq]
        self._current_params = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        """Set up the thermal mixer scene."""
        super()._setup_scene()

    def _get_observations(self) -> dict:
        """Return observations: arm pos + arm vel + gripper pos + ee pos.

        Observation vector (17-dim):
          [0:6]  arm joint positions (rad)
          [6:12] arm joint velocities (rad/s)
          [12:14] gripper joint positions (m)
          [14:17] end-effector position relative to env origin (m)
        """
        arm_pos = self.joint_pos[:, self._arm_dof_indices]
        arm_vel = self.joint_vel[:, self._arm_dof_indices]
        gripper_pos = self.joint_pos[:, self._gripper_dof_indices]

        # End-effector position (tool0 body) relative to env origin
        ee_idx = self.robot.find_bodies("wrist_3_link")[0]
        ee_pos = self.robot.data.body_pos_w[:, ee_idx, :] - self.scene.env_origins.unsqueeze(1)
        ee_pos = ee_pos.squeeze(1)

        obs = torch.cat([arm_pos, arm_vel, gripper_pos, ee_pos], dim=-1)
        return {"policy": obs}

    def _compute_reward_temp(self) -> torch.Tensor:
        """Temperature reward component.

        score = max(0, 1 - |current - target| / |target - 25|)
        """
        current = self._current_params[:, 0]
        target = self._target_params[:, 0]
        denom = torch.abs(target - DEFAULT_TEMP).clamp(min=1.0)
        score = torch.clamp(1.0 - torch.abs(current - target) / denom, min=0.0)
        return score

    def _compute_reward_time(self) -> torch.Tensor:
        """Time reward component.

        score = max(0, 1 - |current - target| / |target - 60|)
        """
        current = self._current_params[:, 1]
        target = self._target_params[:, 1]
        denom = torch.abs(target - DEFAULT_TIME).clamp(min=1.0)
        score = torch.clamp(1.0 - torch.abs(current - target) / denom, min=0.0)
        return score

    def _compute_reward_freq(self) -> torch.Tensor:
        """Frequency reward component.

        score = max(0, 1 - |current - target| / |target - 900|)
        """
        current = self._current_params[:, 2]
        target = self._target_params[:, 2]
        denom = torch.abs(target - DEFAULT_FREQ).clamp(min=1.0)
        score = torch.clamp(1.0 - torch.abs(current - target) / denom, min=0.0)
        return score

    def _get_rewards(self) -> torch.Tensor:
        """Continuous weighted reward.

        Returns:
            (num_envs,) tensor with reward in [0, 1].

        NOTE: Currently returns 0.0 as a placeholder. The full reward
        requires reading the Thermomixer's display state from the physics
        simulation, which depends on a UI state machine not yet implemented.
        """
        # TODO: Update _current_params from simulation state when the
        # mixer's display model is available.

        # Placeholder: return zeros
        return torch.zeros(self.num_envs, device=self.device)

        # --- Full reward (uncomment when display state is available) ---
        # r_temp = self._compute_reward_temp()
        # r_time = self._compute_reward_time()
        # r_freq = self._compute_reward_freq()
        # return (
        #     self.cfg.reward_weight_temp * r_temp
        #     + self.cfg.reward_weight_time * r_time
        #     + self.cfg.reward_weight_freq * r_freq
        # )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Done conditions: timeout only (no early termination)."""
        terminated, time_out = super()._get_dones()
        # early_stop = False: no success-based termination
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset with arm perturbation and random target parameters."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # Randomize arm joint positions with perturbation
        perturbation = sample_uniform(
            self._perturb_low,
            self._perturb_high,
            (num_reset, _ARM_DOF_COUNT),
            self.device,
        )

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos[:, self._arm_dof_indices] += perturbation

        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        # Randomize target parameters
        target_temp = sample_uniform(
            torch.tensor(TARGET_TEMP_LOW, device=self.device),
            torch.tensor(TARGET_TEMP_HIGH, device=self.device),
            (num_reset,),
            self.device,
        )
        target_time = sample_uniform(
            torch.tensor(TARGET_TIME_LOW, device=self.device),
            torch.tensor(TARGET_TIME_HIGH, device=self.device),
            (num_reset,),
            self.device,
        )
        target_freq = sample_uniform(
            torch.tensor(TARGET_FREQ_LOW, device=self.device),
            torch.tensor(TARGET_FREQ_HIGH, device=self.device),
            (num_reset,),
            self.device,
        )

        self._target_params[env_ids, 0] = target_temp
        self._target_params[env_ids, 1] = target_time
        self._target_params[env_ids, 2] = target_freq

        # Reset current mixer state to defaults
        self._current_params[env_ids, 0] = DEFAULT_TEMP
        self._current_params[env_ids, 1] = DEFAULT_TIME
        self._current_params[env_ids, 2] = DEFAULT_FREQ
