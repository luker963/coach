from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.agents.dfp_agent import DFPAgentParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.middleware_parameters import FCMiddlewareParameters
from rl_coach.graph_managers.graph_manager import SimpleSchedule, SimpleScheduleWithoutEvaluation
from rl_coach.core_types import EnvironmentSteps, TrainingSteps, AlwaysDumpFilter, EnvironmentEpisodes
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, MiddlewareScheme, \
    EmbedderScheme
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.environments.first_test import ControlSuiteEnvironmentParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

agent_params = RainbowDQNAgentParameters()

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
