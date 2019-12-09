from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import ActionType
from rl_coach.environments.environment import Environment, LevelSelection
from rl_coach.spaces import PlanarMapsObservationSpace, DiscreteActionSpace, StateSpace, VectorObservationSpace, \
    TensorObservationSpace
import numpy as np
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from typing import Union


class Player:
    x = 0
    y = 0

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y


class FacEnv:
    map = None
    target = None
    player = None
    wrongs = None

    def __init__(self):
        self.wrongs = 0
        self.player = Player()
        # self.target = {'x': 2, 'y': 2}
        self.target = {'x': np.random.randint(-25, 25), 'y': np.random.randint(-25, 25)}
        print(self.target['x'], self.target['y'])
        self.observation = np.zeros((84, 84, 2), dtype=int)
        # setting target
        self.observation[self.target['x']][self.target['y']][1] = 1
        # setting player
        self.observation[self.player.x][self.player.y][0] = 1


class TestCls(Environment):
    env = None
    reward_limit = 0

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters,
                         **kwargs)
        self.state_space = StateSpace({
            "observation": VectorObservationSpace(shape=4)
            # "observation": PlanarMapsObservationSpace(shape=np.array([84, 84, 2]), low=0, high=1)
            # "observation": TensorObservationSpace(shape=np.array([2, 2]), low=0, high=1)
            # "player": VectorObservationSpace(shape=2),
            # "map": PlanarMapsObservationSpace(shape=np.array([84, 84, 1]), low=0, high=2),
        })
        self.goal_space = VectorObservationSpace(shape=4)
        # self.state_space = PlanarMapsObservationSpace(shape=np.array([84, 84, 2]), low=0, high=1)
        self.action_space = DiscreteActionSpace(num_actions=4,
                                                descriptions={"0": "up", "1": "down", "2": "left", "3": "right"})
        self.env = FacEnv()
        self.reward_limit = -np.power(np.sum(np.arange(1 + np.abs(self.env.target['x']) + np.abs(self.env.target['y'])))/2, 2)

    def _take_action(self, action_idx: ActionType) -> None:
        # print(self.env.player.y, self.env.player.x)
        self.env.observation[self.env.player.x][self.env.player.y] = 0
        if action_idx == 0:
            self.env.player.y += 1
        if action_idx == 1:
            self.env.player.y -= 1
        if action_idx == 2:
            self.env.player.x -= 1
        if action_idx == 3:
            self.env.player.x += 1
        self.env.observation[self.env.player.x][self.env.player.y] = 1

    def _update_state(self) -> None:
        new_reward = -(
                    np.abs(self.env.player.x - self.env.target['x']) + np.abs(self.env.player.y - self.env.target['y']))

        if self.reward > new_reward:
            self.env.wrongs += 1
        self.reward = new_reward
        self.goal = np.array([self.env.target['x'], self.env.target['y'], self.env.target['x'], self.env.target['y']])
        # self.done = ((self.env.player.x == self.env.target['x']) and (self.env.player.y == self.env.target['y'])) or (np.abs(self.env.player.x) > 80 or np.abs(self.env.player.y) > 80)
        self.done = (np.abs(self.env.player.x) > 80 or np.abs(
            self.env.player.y) > 80) or self.total_reward_in_current_episode < self.reward_limit or ((self.env.player.x == self.env.target['x']) and (self.env.player.y == self.env.target['y']))
        # self.done = self.current_episode_steps_counter > 2500
        self.state = {
            "observation": np.array([self.env.player.x, self.env.player.y, self.env.target['x'], self.env.target['y']])
            # "observation": self.env.observation
            # "observation": np.array([[self.env.player.x, self.env.player.y], [self.env.target['x'], self.env.target['y']]])
            # "player": np.array([self.env.player.x, self.env.player.y]),
            # "map": self.env.map
        }

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        if (self.env.player.x == self.env.target['x']) and (self.env.player.y == self.env.target['y']):
            print("TARGET")
        if np.abs(self.env.player.x) > 80 or np.abs(self.env.player.y) > 80:
            print("LOST")
        self.env.player = Player()
        self.env.target = {'x': np.random.randint(-25, 25), 'y': np.random.randint(-25, 25)}
        print("Wrongs: ", self.env.wrongs)
        print(self.env.target['x'], self.env.target['y'])
        self.env.wrongs = 0
        self.reward_limit = -np.power(np.sum(np.arange(1 + np.abs(self.env.target['x']) + np.abs(self.env.target['y'])))/2, 2)

    def get_rendered_image(self):
        return self.env.map


class ControlSuiteEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()

    @property
    def path(self):
        return 'rl_coach.environments.first_test:TestCls'
