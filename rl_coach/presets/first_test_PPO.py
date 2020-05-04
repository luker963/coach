from rl_coach.agents.actor_critic_agent import ActorCriticAgentParameters
from rl_coach.agents.ppo_agent import PPOAlgorithmParameters, PPOAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import EnvironmentSteps, AlwaysDumpFilter, EnvironmentEpisodes
from rl_coach.environments.first_test import ControlSuiteEnvironmentParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import SimpleSchedule

agent_params = PPOAgentParameters()
schedule_params = SimpleSchedule()
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.heatup_steps = EnvironmentSteps(2000)
preset_validation_params = PresetValidationParameters()

vis_params = VisualizationParameters(dump_gifs=False, video_dump_methods=AlwaysDumpFilter())

env_params = ControlSuiteEnvironmentParameters()

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=vis_params,
                                    preset_validation_params=preset_validation_params)
