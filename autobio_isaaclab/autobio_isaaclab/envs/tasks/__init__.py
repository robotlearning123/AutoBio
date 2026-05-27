"""AutoBio task environments.

Each task registers a gymnasium environment ID following the pattern:
``Isaac-AutoBio-{TaskName}-Direct-v0``
"""

import gymnasium as gym

from .pickup import PickupEnv, PickupEnvCfg

# Register gymnasium environments
gym.register(
    id="Isaac-AutoBio-Pickup-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.pickup:PickupEnv",
    kwargs={"cfg": PickupEnvCfg()},
    max_episode_steps=int(22.5 * 250 / 10),  # episode_length_s / (dt * decimation)
)
