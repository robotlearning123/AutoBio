"""Vortex mixer task: operate a Vortex Genie 2 mixer.

Ports AutoBio's ``VortexMixer`` task to Isaac Lab DirectRLEnv.

Scene: ``usd_assets/vortex_mixer/vortex_mixer/vortex_mixer.usda``

Dual ALOHA arms (14 DOF):
  arm1: hold / position tube on the mixer platform
  arm2: operate mixer controls (switch + knob)

Instrument: Vortex Genie 2
  - Switch: 3 states (off / on / continuous)
  - Knob:   continuous 0-10 speed dial

Success is placeholder (returns False / 0.0 for now).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
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
_VORTEX_SCENE_USD = os.path.join(_USD_DIR, "vortex_mixer", "vortex_mixer", "vortex_mixer.usda")

# ---------------------------------------------------------------------------
# ALOHA joint definitions (dual-arm, 14 DOF total)
# Each arm: 6 arm joints + 1 gripper DOF
# ---------------------------------------------------------------------------

# Arm 1 (left ALOHA) -- tube handler
ARM1_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
]
GRIPPER1_JOINT_NAMES = ["left_left_finger", "left_right_finger"]

# Arm 2 (right ALOHA) -- mixer controls (USD uses left_*_2 naming)
ARM2_JOINT_NAMES = [
    "left_waist_2",
    "left_shoulder_2",
    "left_elbow_2",
    "left_forearm_roll_2",
    "left_wrist_angle_2",
    "left_wrist_rotate_2",
]
GRIPPER2_JOINT_NAMES = ["left_left_finger_2", "left_right_finger_2"]

# ALOHA stiffness / damping / effort (from aloha.py)
_ALOHA_ARM_STIFFNESS = {
    "waist": 43.0,
    "shoulder": 265.0,
    "elbow": 227.0,
    "forearm_roll": 78.0,
    "wrist_angle": 37.0,
    "wrist_rotate": 10.4,
}
_ALOHA_ARM_DAMPING = {
    "waist": 5.76,
    "shoulder": 20.0,
    "elbow": 18.49,
    "forearm_roll": 6.78,
    "wrist_angle": 6.28,
    "wrist_rotate": 1.2,
}
_ALOHA_ARM_EFFORT = {
    "waist": 35.0,
    "shoulder": 144.0,
    "elbow": 59.0,
    "forearm_roll": 22.0,
    "wrist_angle": 35.0,
    "wrist_rotate": 35.0,
}

_GRIPPER_STIFFNESS = 2000.0
_GRIPPER_DAMPING = 124.0
_GRIPPER_EFFORT = 35.0

# Default joint positions (both arms at rest)
_DEFAULT_JOINT_POS = {
    "left_waist": 0.0,
    "left_shoulder": -0.96,
    "left_elbow": 1.16,
    "left_forearm_roll": 0.0,
    "left_wrist_angle": -0.3,
    "left_wrist_rotate": 0.0,
    "left_left_finger": 0.0084,
    "left_right_finger": 0.0084,
    "left_waist_2": 0.0,
    "left_shoulder_2": -0.96,
    "left_elbow_2": 1.16,
    "left_forearm_roll_2": 0.0,
    "left_wrist_angle_2": -0.3,
    "left_wrist_rotate_2": 0.0,
    "left_left_finger_2": 0.0084,
    "left_right_finger_2": 0.0084,
}

# ---------------------------------------------------------------------------
# Vortex Genie 2 instrument state
# ---------------------------------------------------------------------------

# Switch states
SWITCH_OFF = 0
SWITCH_ON = 1
SWITCH_CONTINUOUS = 2

# Knob range
KNOB_MIN = 0.0
KNOB_MAX = 10.0


@dataclass
class VortexInstrumentState:
    """Internal state of the Vortex Genie 2 instrument.

    This is a simplified placeholder.  Real vortex dynamics (mixing speed,
    vibration amplitude, tube content mixing) are not yet modelled.
    """

    switch_state: int = SWITCH_OFF  # 0=off, 1=on, 2=continuous
    knob_position: float = 0.0      # 0.0 - 10.0

    @property
    def is_running(self) -> bool:
        """Whether the mixer is currently active."""
        return self.switch_state in (SWITCH_ON, SWITCH_CONTINUOUS) and self.knob_position > 0.0

    @property
    def mixing_speed(self) -> float:
        """Simplified mixing speed (placeholder)."""
        if not self.is_running:
            return 0.0
        return self.knob_position / KNOB_MAX


# ---------------------------------------------------------------------------
# Robot config builder
# ---------------------------------------------------------------------------

def _build_vortex_robot_cfg() -> ArticulationCfg:
    """Build ArticulationCfg for the dual-ALOHA vortex mixer scene.

    The scene USD contains both ALOHA arms, the Vortex Genie 2, and a tube.
    """
    actuator_cfgs: list[ImplicitActuatorCfg] = []

    # Arm 1 joints
    for jname_suffix in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"):
        jname = f"left_{jname_suffix}"
        actuator_cfgs.append(
            ImplicitActuatorCfg(
                joint_names_expr=[jname],
                stiffness=_ALOHA_ARM_STIFFNESS[jname_suffix],
                damping=_ALOHA_ARM_DAMPING[jname_suffix],
                effort_limit=_ALOHA_ARM_EFFORT[jname_suffix],
            )
        )
    # Gripper 1
    actuator_cfgs.append(
        ImplicitActuatorCfg(
            joint_names_expr=GRIPPER1_JOINT_NAMES,
            stiffness=_GRIPPER_STIFFNESS,
            damping=_GRIPPER_DAMPING,
            effort_limit=_GRIPPER_EFFORT,
        )
    )

    # Arm 2 joints
    for jname_suffix in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"):
        jname = f"right_{jname_suffix}"
        actuator_cfgs.append(
            ImplicitActuatorCfg(
                joint_names_expr=[jname],
                stiffness=_ALOHA_ARM_STIFFNESS[jname_suffix],
                damping=_ALOHA_ARM_DAMPING[jname_suffix],
                effort_limit=_ALOHA_ARM_EFFORT[jname_suffix],
            )
        )
    # Gripper 2
    actuator_cfgs.append(
        ImplicitActuatorCfg(
            joint_names_expr=GRIPPER2_JOINT_NAMES,
            stiffness=_GRIPPER_STIFFNESS,
            damping=_GRIPPER_DAMPING,
            effort_limit=_GRIPPER_EFFORT,
        )
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_VORTEX_SCENE_USD,
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
            "arm1": ImplicitActuatorCfg(
                joint_names_expr=ARM1_JOINT_NAMES,
                stiffness=43.0,
                damping=5.76,
            ),
            "gripper1": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER1_JOINT_NAMES,
                stiffness=_GRIPPER_STIFFNESS,
                damping=_GRIPPER_DAMPING,
            ),
            "arm2": ImplicitActuatorCfg(
                joint_names_expr=ARM2_JOINT_NAMES,
                stiffness=43.0,
                damping=5.76,
            ),
            "gripper2": ImplicitActuatorCfg(
                joint_names_expr=GRIPPER2_JOINT_NAMES,
                stiffness=_GRIPPER_STIFFNESS,
                damping=_GRIPPER_DAMPING,
            ),
        },
    )


# Module-level default config
VORTEX_ROBOT_CFG = _build_vortex_robot_cfg()


# ---------------------------------------------------------------------------
# Tube rigid body
# ---------------------------------------------------------------------------

def _build_tube_cfg(prim_path: str) -> RigidObjectCfg:
    """Build RigidObjectCfg for the tube on the vortex mixer (references existing prim)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=None,
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

@configclass
class VortexMixerEnvCfg(AutobioDualEnvCfg):
    """Configuration for the vortex mixer task."""

    task_name = "vortex_mixer"
    episode_length_s = 30.0
    early_stop = True

    # Observation: 16 joint pos + 16 joint vel + 3 tube pos + 3 switch/knob state = 38
    observation_space = 38
    action_space = 16  # 8 per arm (6 arm + 2 gripper)

    # Robot: load the vortex mixer scene USD (dual ALOHA + mixer + tube)
    robot_cfg: ArticulationCfg = VORTEX_ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


# ---------------------------------------------------------------------------
# Env implementation
# ---------------------------------------------------------------------------

class VortexMixerEnv(AutobioDualEnv):
    """Vortex mixer task environment.

    Task: arm1 holds a tube on the mixer platform while arm2 operates
    the Vortex Genie 2 controls (3-state switch + speed knob).

    Instrument state is tracked internally but vortex dynamics are not
    yet modelled (placeholder).
    """

    cfg: VortexMixerEnvCfg

    def __init__(self, cfg: VortexMixerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Per-env instrument state (simplified -- not differentiable)
        self._instrument = [VortexInstrumentState() for _ in range(self.num_envs)]

    def _setup_scene(self):
        """Set up the vortex mixer scene."""
        super()._setup_scene()

        # Tube body: separate rigid body for tracking
        tube_prim = self.cfg.robot_cfg.prim_path.replace(
            "/Robot", "/Robot/Geometry/centrifuge_15ml_screw_body/tn__1_centrifuge_15ml_body_"
        )
        self.tube_body = RigidObject(_build_tube_cfg(tube_prim))
        self.scene.rigid_objects["tube_body"] = self.tube_body

    def _get_observations(self) -> dict:
        """Return observations: joint pos + joint vel + tube pos + instrument state."""
        joint_pos = self.joint_pos[:, :16]
        joint_vel = self.joint_vel[:, :16]

        # Tube position relative to env origin
        tube_pos = self.tube_body.data.root_pos_w[:, :3] - self.scene.env_origins

        # Instrument state as observation (switch one-hot + knob value)
        switch_onehot = torch.zeros(self.num_envs, 3, device=self.device)
        knob_vals = torch.zeros(self.num_envs, 1, device=self.device)
        for i, inst in enumerate(self._instrument):
            switch_onehot[i, inst.switch_state] = 1.0
            knob_vals[i, 0] = inst.knob_position / KNOB_MAX

        instrument_obs = torch.cat([switch_onehot, knob_vals], dim=-1)  # (num_envs, 4)

        obs = torch.cat([joint_pos, joint_vel, tube_pos, instrument_obs], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Placeholder reward: returns 0.0 for now.

        TODO: Implement reward based on:
          - Tube placed on mixer platform (proximity check)
          - Switch toggled to ON/CONTINUOUS
          - Knob set to target speed
          - Mixing duration sufficient
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

        # Reset instrument states
        for i in env_ids:
            self._instrument[i] = VortexInstrumentState()

        # Randomize arm joint positions with small perturbation
        num_reset = len(env_ids)
        perturb_low = torch.full((16,), -0.1, device=self.device)
        perturb_high = torch.full((16,), 0.1, device=self.device)
        perturbation = sample_uniform(perturb_low, perturb_high, (num_reset, 16), self.device)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos += perturbation

        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

    # --- Instrument control helpers (for future policy integration) ---

    def set_switch(self, env_ids: Sequence[int], state: int) -> None:
        """Set the vortex mixer switch state for given environments.

        Args:
            env_ids: Environment indices.
            state: SWITCH_OFF (0), SWITCH_ON (1), or SWITCH_CONTINUOUS (2).
        """
        state = max(SWITCH_OFF, min(SWITCH_CONTINUOUS, state))
        for i in env_ids:
            self._instrument[i].switch_state = state

    def set_knob(self, env_ids: Sequence[int], value: float) -> None:
        """Set the vortex mixer knob position for given environments.

        Args:
            env_ids: Environment indices.
            value: Knob position, clamped to [0.0, 10.0].
        """
        value = max(KNOB_MIN, min(KNOB_MAX, value))
        for i in env_ids:
            self._instrument[i].knob_position = value
