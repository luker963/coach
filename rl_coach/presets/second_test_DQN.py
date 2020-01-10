from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters
from rl_coach.core_types import EnvironmentSteps
from rl_coach.environments.second_test import ControlSuiteEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.non_episodic import ExperienceReplayParameters

experience_replay_parameters = ExperienceReplayParameters()
experience_replay_parameters.max_size = (MemoryGranularity.Transitions, 100000)
agent_params = DQNAgentParameters()
agent_params.memory = experience_replay_parameters
schedule_params = SimpleSchedule()
schedule_params.heatup_steps = EnvironmentSteps(10000)
preset_validation_params = PresetValidationParameters(test=True, min_reward_threshold=-50000, max_episodes_to_achieve_reward=10, num_workers=5)
vis_params = VisualizationParameters(render=False)

env_params = ControlSuiteEnvironmentParameters()
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
