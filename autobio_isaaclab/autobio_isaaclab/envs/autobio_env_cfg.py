"""Base configuration for AutoBio Isaac Lab environments."""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab.assets import ArticulationCfg
from isaaclab_physx.physics import PhysxCfg

from ..assets.aloha import ALOHA_CFG


@configclass
class AutobioEnvCfg(DirectRLEnvCfg):
    """Base configuration for all AutoBio tasks.

    Matches AutoBio's simulation parameters:
    - MuJoCo timestep: 0.002s (500 Hz)
    - Policy rate: ~20 Hz (decimation=10 gives 50 Hz at dt=1/500, or 25 Hz at dt=1/250)
    """

    # Simulation
    decimation = 10
    episode_length_s = 30.0
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 250,
        render_interval=decimation,
        physics=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=64,
            max_velocity_iteration_count=16,
            gpu_max_rigid_contact_count=2**20,
        ),
    )

    # Spaces
    action_space = 7  # 6 DOF arm + 1 gripper
    observation_space = 7  # joint positions
    state_space = 0

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    # Robot
    robot_cfg: ArticulationCfg = ALOHA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Task-specific fields (overridden by subclasses)
    time_limit: float = 30.0
    early_stop: bool = True
    task_name: str = "base"
