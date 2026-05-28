"""UR5e + Robotiq 2F-85 gripper asset configuration for Isaac Lab.

The UR5e is a 6-DOF industrial arm with a Robotiq 2F-85 parallel-jaw gripper.
Joint structure: shoulder_pan (Z), shoulder_lift (Y), elbow (Y),
wrist_1 (Y), wrist_2 (Z), wrist_3 (Y), right_driver_joint (X),
left_driver_joint (X).

Unlike ALOHA, the UR5e is always loaded as part of a scene USD (compositional
MJCF), not as a standalone robot USD. Tasks override ``usd_path`` to point at
their specific scene file.

Source: ``AutoBio/autobio/model/robot/ur5e/`` + ``AutoBio/autobio/model/robot/robotiq/``
"""

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

# Path to converted USD assets (populated by convert_assets.py)
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "usd_assets")

# Default scene USD -- tasks should override this path
_DEFAULT_SCENE_USD = os.path.join(_USD_DIR, "insert", "insert", "insert.usda")

# UR5e arm joint names (6-DOF, matching MJCF naming)
UR5E_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]

# Robotiq 2F-85 gripper joint names (all linkage joints in the USD)
UR5E_GRIPPER_JOINT_NAMES = [
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
UR5E_JOINT_NAMES = UR5E_ARM_JOINT_NAMES + UR5E_GRIPPER_JOINT_NAMES

# Per-joint stiffness (kp) from MuJoCo defaults
UR5E_ARM_STIFFNESS = {
    "shoulder_pan": 100.0,
    "shoulder_lift": 100.0,
    "elbow": 100.0,
    "wrist_1": 100.0,
    "wrist_2": 100.0,
    "wrist_3": 100.0,
}

# Per-joint damping (kv) from MuJoCo defaults
UR5E_ARM_DAMPING = {
    "shoulder_pan": 10.0,
    "shoulder_lift": 10.0,
    "elbow": 10.0,
    "wrist_1": 10.0,
    "wrist_2": 10.0,
    "wrist_3": 10.0,
}

# Effort limits from UR5e datasheet (Nm)
UR5E_ARM_EFFORT = {
    "shoulder_pan": 150.0,
    "shoulder_lift": 150.0,
    "elbow": 150.0,
    "wrist_1": 28.0,
    "wrist_2": 28.0,
    "wrist_3": 28.0,
}

# Default initial joint positions (home pose -- arm slightly raised)
UR5E_DEFAULT_JOINT_POS = {
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


def make_ur5e_2f85_cfg(
    usd_path: str | None = None,
    prim_path: str = "/World/envs/env_.*/Robot",
) -> ArticulationCfg:
    """Build a UR5e + 2F-85 ArticulationCfg.

    Args:
        usd_path: Path to the scene USD containing the UR5e robot.
            Defaults to the ``insert`` scene if not provided.
        prim_path: Prim path pattern for the articulation root.

    Returns:
        Configured ArticulationCfg for the UR5e + 2F-85 robot.
    """
    scene_usd = usd_path or _DEFAULT_SCENE_USD

    actuators = {}
    for jname in UR5E_ARM_JOINT_NAMES:
        actuators[jname] = ImplicitActuatorCfg(
            joint_names_expr=[jname],
            stiffness=UR5E_ARM_STIFFNESS[jname],
            damping=UR5E_ARM_DAMPING[jname],
            effort_limit=UR5E_ARM_EFFORT[jname],
        )
    actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=UR5E_GRIPPER_JOINT_NAMES,
        stiffness=2000.0,
        damping=100.0,
        effort_limit=20.0,
    )

    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=scene_usd,
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
            joint_pos=UR5E_DEFAULT_JOINT_POS,
        ),
        actuators=actuators,
        prim_path=prim_path,
    )

    # Store per-joint metadata for runtime use
    cfg.metadata = {
        "stiffness_map": dict(UR5E_ARM_STIFFNESS),
        "damping_map": dict(UR5E_ARM_DAMPING),
        "effort_map": dict(UR5E_ARM_EFFORT),
        "joint_names": list(UR5E_JOINT_NAMES),
    }

    return cfg


# Default UR5E + 2F-85 configuration (uses insert scene as placeholder)
UR5E_2F85_CFG = make_ur5e_2f85_cfg()
"""Default ArticulationCfg for the UR5e + Robotiq 2F-85 robot."""
