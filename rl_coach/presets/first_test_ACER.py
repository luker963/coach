from rl_coach.agents.acer_agent import ACERAgentParameters
from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import EnvironmentSteps, AlwaysDumpFilter, EnvironmentEpisodes
from rl_coach.environments.first_test import ControlSuiteEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.memories.memory import MemoryGranularity

agent_params = ACERAgentParameters()
agent_params.algorithm.num_steps_between_gradient_updates = 300
agent_params.algorithm.apply_gradients_every_x_episodes = 10
agent_params.network_wrappers['main'].learning_rate = 0.000001
agent_params.algorithm.ratio_of_replay = 4
agent_params.algorithm.num_transitions_to_start_replay = 2000
agent_params.memory.max_size = (MemoryGranularity.Transitions, 100000)
agent_params.algorithm.beta_entropy = 0.01
agent_params.network_wrappers['main'].clip_gradients = 40.
schedule_params = SimpleSchedule()
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.heatup_steps = EnvironmentSteps(2000)
preset_validation_params = PresetValidationParameters()

vis_params = VisualizationParameters(dump_gifs=False, video_dump_methods=AlwaysDumpFilter())

env_params = ControlSuiteEnvironmentParameters()

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
