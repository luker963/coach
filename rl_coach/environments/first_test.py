from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import ActionType
from rl_coach.environments.environment import Environment, LevelSelection
from rl_coach.spaces import PlanarMapsObservationSpace, DiscreteActionSpace, StateSpace, VectorObservationSpace, \
    TensorObservationSpace, ImageObservationSpace
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
    target = None
    wrongs = None

    def __init__(self):
        self.wrongs = 0
        # self.target = {'x': 2, 'y': 2}
        self.target = {'x': np.random.randint(1, 14), 'y': np.random.randint(1, 14)}
        print(self.target['x'], self.target['y'])
        self.observation = np.zeros((16, 16, 1), dtype=int)
        # setting target
        self.observation[self.target['x']][self.target['y']][0] = 1


class TestCls(Environment):
    env = None
    new_reward = 0

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters,
                         **kwargs)
        self.state_space = StateSpace({
            # "observation": VectorObservationSpace(shape=2)
            "observation": PlanarMapsObservationSpace(shape=np.array([16, 16, 1]), low=0, high=1)
            # "observation": TensorObservationSpace(shape=np.array([2, 2]), low=0, high=1)
            # "player": VectorObservationSpace(shape=2),
            # "map": PlanarMapsObservationSpace(shape=np.array([84, 84, 1]), low=0, high=2),
        })
        # self.goal_space = VectorObservationSpace(shape=4)
        # self.state_space = PlanarMapsObservationSpace(shape=np.array([84, 84, 2]), low=0, high=1)
        self.action_space = DiscreteActionSpace(num_actions=4,
                                                descriptions={"0": "up", "1": "down", "2": "left", "3": "right"})
        self.env = FacEnv()

    def _take_action(self, action_idx: ActionType) -> None:
        self.env.observation[self.env.target['x']][self.env.target['y']][0] = 0
        if action_idx == 0:
            self.env.target['y'] += 1
        if action_idx == 1:
            self.env.target['y'] -= 1
        if action_idx == 2:
            self.env.target['x'] -= 1
        if action_idx == 3:
            self.env.target['x'] += 1
        # self.new_reward = -(np.abs(7 - self.env.target['x']) + np.abs(7 - self.env.target['y']))
        if self.env.target['x'] == 0:
            self.done = True
            self.reward = -10000
            self.env.target['x'] = 1
            # self.new_reward *= 5
        if self.env.target['y'] == 0:
            self.done = True
            self.reward = -10000
            self.env.target['y'] = 1
            # self.new_reward *= 5
        if self.env.target['x'] == 15:
            self.done = True
            self.reward = -10000
            self.env.target['x'] = 14
            # self.new_reward *= 5
        if self.env.target['y'] == 15:
            self.done = True
            self.reward = -10000
            self.env.target['y'] = 14
            # self.new_reward *= 5
        self.env.observation[self.env.target['x']][self.env.target['y']][0] = 1
        print("Moved to: ", self.env.target['x'], ", ", self.env.target['y'])

    def _update_state(self) -> None:
        if not self.done:
            if self.reward > self.new_reward:
                self.env.wrongs += 1
            if self.env.target['x'] == 7 and self.env.target['y'] == 7:
                self.new_reward = 10000
            self.reward = -(np.abs(7 - self.env.target['x']) + np.abs(7 - self.env.target['y']))
            # self.goal = np.array([self.env.target['x'], self.env.target['y'], self.env.target['x'], self.env.target['y']])
            # self.done = ((self.env.player.x == self.env.target['x']) and (self.env.player.y == self.env.target['y'])) or (np.abs(self.env.player.x) > 80 or np.abs(self.env.player.y) > 80)
            self.done = (self.env.target['x'] == 7 and self.env.target['y'] == 7) or self.env.target['x'] <= 0 or self.env.target['y'] <= 0 or self.env.target['x'] >= 15 or self.env.target['y'] >= 15 or self.current_episode_steps_counter > 125
        self.state = {
            # "observation": np.array([self.env.target['x'], self.env.target['y']])
            "observation": self.env.observation
            # "observation": np.array([[self.env.player.x, self.env.player.y], [self.env.target['x'], self.env.target['y']]])
            # "player": np.array([self.env.player.x, self.env.player.y]),
            # "map": self.env.map
        }

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        if (7 == self.env.target['x']) and (7 == self.env.target['y']):
            print("TARGET")
        if self.env.target['x'] > 15 or self.env.target['y'] > 15 or self.env.target['x'] < 1 or self.env.target['y'] < 1:
            print("LOST")
        self.env.target = {'x': np.random.randint(1, 14), 'y': np.random.randint(1, 14)}
        # self.env.target = {'x': 34, 'y': 12}
        self.env.observation = np.zeros((16, 16, 1), dtype=int)
        self.env.observation[self.env.target['x']][self.env.target['y']][0] = 1
        print("Wrongs: ", self.env.wrongs)
        print(self.env.target['x'], self.env.target['y'])
        self.env.wrongs = 0

    def get_rendered_image(self):
        image_map = np.copy(self.env.observation[:, :, 0]).astype(float)
        image_map[image_map == 0] = 255
        image_map[image_map == 1] = 170
        # new_image = np.zeros((50, 50, 3))
        # image_map = np.expand_dims(image_map, 2)
        # new_image[:, :, :] = image_map[:, :]
        # new_image[new_image == 1] = (255, 0, 0)
        # image_map[image_map == 0] = 50
        # image_map[self.player.x, self.player.y] = [0, 255, 0]
        return image_map


class ControlSuiteEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()

    @property
    def path(self):
        return 'rl_coach.environments.first_test:TestCls'
