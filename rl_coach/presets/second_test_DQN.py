from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager

from rl_coach.base_parameters import PresetValidationParameters, VisualizationParameters

from rl_coach.core_types import EnvironmentSteps
from rl_coach.environments.second_test import ControlSuiteEnvironmentParameters

from rl_coach.graph_managers.graph_manager import SimpleSchedule

from rl_coach.agents.dqn_agent import DQNAgentParameters

agent_params = DQNAgentParameters()
schedule_params = SimpleSchedule()
schedule_params.heatup_steps = EnvironmentSteps(10)
preset_validation_params = PresetValidationParameters()
vis_params = VisualizationParameters(render=False)

env_params = ControlSuiteEnvironmentParameters()
graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
