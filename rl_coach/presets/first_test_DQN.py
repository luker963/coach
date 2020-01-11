from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.graph_managers.graph_manager import SimpleSchedule, SimpleScheduleWithoutEvaluation
from rl_coach.core_types import EnvironmentSteps, TrainingSteps, AlwaysDumpFilter, EnvironmentEpisodes
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, MiddlewareScheme, \
    EmbedderScheme
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.environments.first_test import ControlSuiteEnvironmentParameters
from rl_coach.schedules import LinearSchedule

agent_params = DQNAgentParameters()
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
agent_params.exploration.evaluation_epsilon = 0
agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0.1, 100000)
# agent_params.network_wrappers['main'].middleware_parameters = FCMiddlewareParameters(scheme=MiddlewareScheme.Medium)
# agent_params.network_wrappers['main'].batch_size = 1
agent_params.network_wrappers['main'].should_get_softmax_probabilities = True
agent_params.network_wrappers['main'].input_embedders_parameters = {
    "observation": InputEmbedderParameters(scheme=EmbedderScheme.Empty)
}
# agent_params.network_wrappers['main'].input_embedders_parameters['measurements'] = agent_params.network_wrappers['main'].input_embedders_parameters.pop('observation')
schedule_params = SimpleSchedule()
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(50)
schedule_params.heatup_steps = EnvironmentSteps(0)
preset_validation_params = PresetValidationParameters()
# preset_validation_params.test = True
# preset_validation_params.min_reward_threshold = 20
# preset_validation_params.max_episodes_to_achieve_reward = 400

vis_params = VisualizationParameters(dump_gifs=True, video_dump_methods=AlwaysDumpFilter())

env_params = ControlSuiteEnvironmentParameters()


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)