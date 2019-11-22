from rl_coach.agents.acer_agent import ACERAgentParameters
from rl_coach.memories.memory import MemoryGranularity

from rl_coach.memories.non_episodic import ExperienceReplayParameters

from rl_coach.exploration_policies.e_greedy import EGreedyParameters

from rl_coach.agents.dqn_agent import DQNAgentParameters, DQNAlgorithmParameters, DQNNetworkParameters

from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters, AgentParameters
from rl_coach.core_types import EnvironmentSteps, EnvironmentEpisodes
from rl_coach.environments.second_test import ControlSuiteEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule

schedule_params = SimpleSchedule()
# schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
# schedule_params.evaluation_steps = EnvironmentEpisodes(1)
# schedule_params.heatup_steps = EnvironmentSteps(10)

agent_params = ACERAgentParameters()
agent_params.memory.max_size = (MemoryGranularity.Transitions, 50000)
agent_params.algorithm.apply_gradients_every_x_episodes = 1
agent_params.algorithm.num_steps_between_gradient_updates = 20
agent_params.algorithm.beta_entropy = 0.05
# agent_params.memory.max_size = (MemoryGranularity.Transitions, 5000)
# agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(100)
# agent_params.algorithm.num_consecutive_playing_steps = EnvironmentSteps(1)

preset_validation_params = PresetValidationParameters(test=False, min_reward_threshold=-50000, max_episodes_to_achieve_reward=10)
vis_params = VisualizationParameters(render=False)

env_params = ControlSuiteEnvironmentParameters()
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
