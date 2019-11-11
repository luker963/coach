from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters

from rl_coach.memories.memory import MemoryGranularity

from rl_coach.memories.non_episodic import ExperienceReplayParameters

from rl_coach.exploration_policies.e_greedy import EGreedyParameters

from rl_coach.agents.dqn_agent import DQNAgentParameters, DQNAlgorithmParameters, DQNNetworkParameters

from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters, AgentParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.environments.second_test import ControlSuiteEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule

agent_params = ActorCriticAgentParameters()
schedule_params = SimpleSchedule()
preset_validation_params = PresetValidationParameters(test=True, min_reward_threshold=-50000, max_episodes_to_achieve_reward=10, num_workers=5)
vis_params = VisualizationParameters(render=False)

env_params = ControlSuiteEnvironmentParameters()
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
