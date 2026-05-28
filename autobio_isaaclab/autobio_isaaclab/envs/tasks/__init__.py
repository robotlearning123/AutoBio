"""AutoBio task environments.

Each task registers a gymnasium environment ID following the pattern:
``Isaac-AutoBio-{TaskName}-Direct-v0``
"""

import gymnasium as gym

from .centrifuge_5430 import Centrifuge5430Env, Centrifuge5430EnvCfg
from .centrifuge_5910 import Centrifuge5910Env, Centrifuge5910EnvCfg
from .centrifuge_mini import CentrifugeMiniEnv, CentrifugeMiniEnvCfg
from .insert import InsertEnv, InsertEnvCfg
from .pickup import PickupEnv, PickupEnvCfg
from .pipette import PipetteEnv, PipetteEnvCfg
from .screw_loose import ScrewLooseEnv, ScrewLooseEnvCfg
from .screw_tighten import ScrewTightenEnv, ScrewTightenEnvCfg
from .thermal_cycler import ThermalCyclerEnv, ThermalCyclerEnvCfg
from .thermal_mixer import ThermalMixerEnv, ThermalMixerEnvCfg
from .vortex_mixer import VortexMixerEnv, VortexMixerEnvCfg

# Register gymnasium environments
gym.register(
    id="Isaac-AutoBio-Pickup-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.pickup:PickupEnv",
    kwargs={"cfg": PickupEnvCfg()},
    max_episode_steps=int(22.5 * 250 / 10),  # episode_length_s / (dt * decimation)
)

gym.register(
    id="Isaac-AutoBio-Insert-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.insert:InsertEnv",
    kwargs={"cfg": InsertEnvCfg()},
    max_episode_steps=int(15.0 * 250 / 10),  # episode_length_s / (dt * decimation)
)

gym.register(
    id="Isaac-AutoBio-ThermalMixer-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.thermal_mixer:ThermalMixerEnv",
    kwargs={"cfg": ThermalMixerEnvCfg()},
    max_episode_steps=int(30.0 * 250 / 10),  # episode_length_s / (dt * decimation)
)

gym.register(
    id="Isaac-AutoBio-Pipette-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.pipette:PipetteEnv",
    kwargs={"cfg": PipetteEnvCfg()},
    max_episode_steps=int(30.0 * 250 / 10),
)

gym.register(
    id="Isaac-AutoBio-VortexMixer-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.vortex_mixer:VortexMixerEnv",
    kwargs={"cfg": VortexMixerEnvCfg()},
    max_episode_steps=int(30.0 * 250 / 10),
)

gym.register(
    id="Isaac-AutoBio-Centrifuge5430-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.centrifuge_5430:Centrifuge5430Env",
    kwargs={"cfg": Centrifuge5430EnvCfg()},
    max_episode_steps=int(15.0 * 250 / 10),
)

gym.register(
    id="Isaac-AutoBio-Centrifuge5910-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.centrifuge_5910:Centrifuge5910Env",
    kwargs={"cfg": Centrifuge5910EnvCfg()},
    max_episode_steps=int(30.0 * 250 / 10),
)

gym.register(
    id="Isaac-AutoBio-CentrifugeMini-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.centrifuge_mini:CentrifugeMiniEnv",
    kwargs={"cfg": CentrifugeMiniEnvCfg()},
    max_episode_steps=int(30.0 * 250 / 10),
)

# Thermal cycler close task
_close_cfg = ThermalCyclerEnvCfg(task_variant="thermal_cycler_close")
gym.register(
    id="Isaac-AutoBio-ThermalCyclerClose-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.thermal_cycler:ThermalCyclerEnv",
    kwargs={"cfg": _close_cfg},
    max_episode_steps=int(31.5 * 250 / 10),  # episode_length_s / (dt * decimation)
)

# Thermal cycler open task
_open_cfg = ThermalCyclerEnvCfg(task_variant="thermal_cycler_open")
gym.register(
    id="Isaac-AutoBio-ThermalCyclerOpen-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.thermal_cycler:ThermalCyclerEnv",
    kwargs={"cfg": _open_cfg},
    max_episode_steps=int(22.5 * 250 / 10),  # episode_length_s / (dt * decimation)
)

gym.register(
    id="Isaac-AutoBio-ScrewLoose-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.screw_loose:ScrewLooseEnv",
    kwargs={"cfg": ScrewLooseEnvCfg()},
    max_episode_steps=int(41.0 * 250 / 10),  # episode_length_s / (dt * decimation)
)

gym.register(
    id="Isaac-AutoBio-ScrewTighten-Direct-v0",
    entry_point="autobio_isaaclab.envs.tasks.screw_tighten:ScrewTightenEnv",
    kwargs={"cfg": ScrewTightenEnvCfg()},
    max_episode_steps=int(45.0 * 250 / 10),  # episode_length_s / (dt * decimation)
)
