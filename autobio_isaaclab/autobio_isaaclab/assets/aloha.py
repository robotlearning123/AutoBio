"""ALOHA robot arm asset configuration for Isaac Lab.

The ALOHA arm is a 6-DOF ViperX 300S with a parallel-jaw gripper.
Joint structure: waist (Z), shoulder (Y), elbow (Y), forearm_roll (X),
wrist_angle (Y), wrist_rotate (X), left_finger (slide), right_finger (slide).

Source: ``AutoBio/autobio/model/robot/aloha_left.xml``
"""

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.sim import schemas
from isaaclab.utils import configclass

import isaaclab.sim as sim_utils

# Path to converted USD assets (populated by convert_assets.py)
_USD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "usd_assets")
_ALOHA_USD = os.path.join(_USD_DIR, "aloha_left", "aloha_left.usd")

# ALOHA joint names (matching MJCF naming with prefix)
ALOHA_JOINT_NAMES = [
    "left/waist",
    "left/shoulder",
    "left/elbow",
    "left/forearm_roll",
    "left/wrist_angle",
    "left/wrist_rotate",
    "left/left_finger",
    "left/right_finger",
]

# Actuator group definitions with kp/kv from MJCF defaults
ALOHA_ACTUATOR_GROUPS = {
    # Arm joints - position control
    "arm": {
        "joint_names_expr": [
            "left/waist",
            "left/shoulder",
            "left/elbow",
            "left/forearm_roll",
            "left/wrist_angle",
            "left/wrist_rotate",
        ],
        "stiffness": {
            "left/waist": 43.0,
            "left/shoulder": 265.0,
            "left/elbow": 227.0,
            "left/forearm_roll": 78.0,
            "left/wrist_angle": 37.0,
            "left/wrist_rotate": 10.4,
        },
        "damping": {
            "left/waist": 5.76,
            "left/shoulder": 20.0,
            "left/elbow": 18.49,
            "left/forearm_roll": 6.78,
            "left/wrist_angle": 6.28,
            "left/wrist_rotate": 1.2,
        },
        "effort_limit": {
            "left/waist": 35.0,
            "left/shoulder": 144.0,
            "left/elbow": 59.0,
            "left/forearm_roll": 22.0,
            "left/wrist_angle": 35.0,
            "left/wrist_rotate": 35.0,
        },
    },
    # Gripper joints - position control with high stiffness
    "gripper": {
        "joint_names_expr": ["left/left_finger", "left/right_finger"],
        "stiffness": 2000.0,
        "damping": 124.0,
        "effort_limit": 35.0,
    },
}


def _build_aloha_cfg() -> ArticulationCfg:
    """Build the ALOHA ArticulationCfg from joint definitions."""
    from isaaclab.actuators import ImplicitActuatorCfg

    actuator_cfgs = []

    for group_name, group_def in ALOHA_ACTUATOR_GROUPS.items():
        joint_names = group_def["joint_names_expr"]
        stiffness = group_def["stiffness"]
        damping = group_def["damping"]
        effort = group_def["effort_limit"]

        if isinstance(stiffness, dict):
            # Per-joint stiffness/damping
            for jname in joint_names:
                actuator_cfgs.append(
                    ImplicitActuatorCfg(
                        joint_names_expr=[jname],
                        stiffness=stiffness[jname],
                        damping=damping[jname],
                        effort_limit=effort[jname],
                    )
                )
        else:
            actuator_cfgs.append(
                ImplicitActuatorCfg(
                    joint_names_expr=joint_names,
                    stiffness=stiffness,
                    damping=damping,
                    effort_limit=effort,
                )
            )

    cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_ALOHA_USD,
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
            joint_pos={
                "left/waist": 0.0,
                "left/shoulder": -0.96,
                "left/elbow": 1.16,
                "left/forearm_roll": 0.0,
                "left/wrist_angle": -0.3,
                "left/wrist_rotate": 0.0,
                "left/left_finger": 0.0084,
                "left/right_finger": 0.0084,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "left/waist",
                    "left/shoulder",
                    "left/elbow",
                    "left/forearm_roll",
                    "left/wrist_angle",
                    "left/wrist_rotate",
                ],
                stiffness=43.0,  # Default, overridden per-joint below
                damping=5.76,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["left/left_finger", "left/right_finger"],
                stiffness=2000.0,
                damping=124.0,
            ),
        },
    )

    # Override per-joint stiffness/damping for arm group
    arm_actuator = cfg.actuators["arm"]
    stiffness_map = {
        "left/waist": 43.0,
        "left/shoulder": 265.0,
        "left/elbow": 227.0,
        "left/forearm_roll": 78.0,
        "left/wrist_angle": 37.0,
        "left/wrist_rotate": 10.4,
    }
    damping_map = {
        "left/waist": 5.76,
        "left/shoulder": 20.0,
        "left/elbow": 18.49,
        "left/forearm_roll": 6.78,
        "left/wrist_angle": 6.28,
        "left/wrist_rotate": 1.2,
    }
    effort_map = {
        "left/waist": 35.0,
        "left/shoulder": 144.0,
        "left/elbow": 59.0,
        "left/forearm_roll": 22.0,
        "left/wrist_angle": 35.0,
        "left/wrist_rotate": 35.0,
    }

    # Store as metadata for runtime use
    cfg.metadata = {
        "stiffness_map": stiffness_map,
        "damping_map": damping_map,
        "effort_map": effort_map,
    }

    return cfg


# Default ALOHA configuration
ALOHA_CFG = _build_aloha_cfg()
"""Default ArticulationCfg for the ALOHA robot arm."""
