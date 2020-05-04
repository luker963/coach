from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.agents.qr_dqn_agent import QuantileRegressionDQNAgentParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.layers import Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentSteps, AlwaysDumpFilter, EnvironmentEpisodes
from rl_coach.environments.first_test import ControlSuiteEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

agent_params = QuantileRegressionDQNAgentParameters()
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
agent_params.exploration.evaluation_epsilon = 0
agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0, 1000000)
agent_params.network_wrappers['main'].middleware_parameters.scheme = [Dense(128), Dense(512), Dense(256)]
# agent_params.network_wrappers['main'].middleware_parameters.num_streams = 3
agent_params.network_wrappers['main'].input_embedders_parameters = {
    "observation": InputEmbedderParameters(scheme=EmbedderScheme.Empty)
    # "targets": InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    # "miners": InputEmbedderParameters(scheme=EmbedderScheme.Empty),
    # "chests": InputEmbedderParameters(scheme=EmbedderScheme.Empty)
}
# agent_params.network_wrappers['main'].replace_mse_with_huber_loss = False
# agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(4)
# agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(500)
agent_params.memory.max_size = (MemoryGranularity.Transitions, 100000)
schedule_params = SimpleSchedule()
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(50)
schedule_params.heatup_steps = EnvironmentSteps(20000)
preset_validation_params = PresetValidationParameters()
# preset_validation_params.test = True
# preset_validation_params.min_reward_threshold = 20
# preset_validation_params.max_episodes_to_achieve_reward = 400

vis_params = VisualizationParameters(dump_gifs=False, video_dump_methods=AlwaysDumpFilter())

env_params = ControlSuiteEnvironmentParameters()


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)