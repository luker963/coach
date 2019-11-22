from rl_coach.agents.ddpg_agent import DDPGAgentParameters
from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.graph_managers.graph_manager import SimpleSchedule, SimpleScheduleWithoutEvaluation
from rl_coach.core_types import EnvironmentSteps, TrainingSteps
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.environments.first_test import ControlSuiteEnvironmentParameters
from rl_coach.memories.memory import MemoryGranularity

agent_params = DQNAgentParameters()
# rename the input embedder key from 'observation' to 'measurements'
# agent_params.network_wrappers['main'].input_embedders_parameters['measurements'] = agent_params.network_wrappers['main'].input_embedders_parameters.pop('observation')
schedule_params = SimpleSchedule()
schedule_params.heatup_steps = EnvironmentSteps(10)
preset_validation_params = PresetValidationParameters()
# preset_validation_params.test = True
# preset_validation_params.min_reward_threshold = 20
# preset_validation_params.max_episodes_to_achieve_reward = 400
agent_params.memory.max_size = (MemoryGranularity.Transitions, 5000)

vis_params = VisualizationParameters(render=False)

env_params = ControlSuiteEnvironmentParameters()


graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)