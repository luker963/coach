from rl_coach.agents.pal_agent import PALAgentParameters
from rl_coach.agents.pal_agent import PALAgentParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentSteps, EveryNEpisodesDumpFilter, AlwaysDumpFilter
from rl_coach.environments.second_test import ControlSuiteEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.memories.non_episodic import ExperienceReplayParameters

experience_replay_parameters = ExperienceReplayParameters()
experience_replay_parameters.max_size = (MemoryGranularity.Transitions, 10000)
agent_params = PALAgentParameters()
agent_params.network_wrappers['main'].input_embedders_parameters['player'] = InputEmbedderParameters()
# agent_params.memory = experience_replay_parameters
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = EmbedderScheme.Shallow
schedule_params = SimpleSchedule()
schedule_params.heatup_steps = EnvironmentSteps(100000)
preset_validation_params = PresetValidationParameters(test=True, min_reward_threshold=-50000, max_episodes_to_achieve_reward=10, num_workers=5)
vis_params = VisualizationParameters(render=False, dump_gifs=False, video_dump_methods=AlwaysDumpFilter())
env_params = ControlSuiteEnvironmentParameters()
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
