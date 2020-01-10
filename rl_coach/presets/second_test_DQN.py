from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters

from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters, EmbedderScheme, \
    MiddlewareScheme
from rl_coach.core_types import EveryNEpisodesDumpFilter, AlwaysDumpFilter, EnvironmentSteps
from rl_coach.environments.second_test import ControlSuiteEnvironmentParameters
from rl_coach.exploration_policies.boltzmann import BoltzmannParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule


agent_params = DQNAgentParameters()
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Shallow
agent_params.network_wrappers['main'].input_embedders_parameters['player'] = InputEmbedderParameters()
agent_params.memory.max_size = (MemoryGranularity.Transitions, 10000)
agent_params.network_wrappers['main'].batch_size = 32
# agent_params.exploration = BoltzmannParameters()
# agent_params.exploration.temperature_schedule = LinearSchedule(1, 0.1, 10000)
# agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0.1, 200000)
# agent_params.network_wrappers['main'].learning_rate = 0.0001
schedule_params = SimpleSchedule()
schedule_params.heatup_steps = EnvironmentSteps(50000)
preset_validation_params = PresetValidationParameters(test=False)
vis_params = VisualizationParameters(dump_gifs=True, video_dump_methods=AlwaysDumpFilter())

env_params = ControlSuiteEnvironmentParameters()
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
