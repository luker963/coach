from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import ActionType
from rl_coach.environments.environment import Environment, LevelSelection
from rl_coach.spaces import PlanarMapsObservationSpace, DiscreteActionSpace, StateSpace, VectorObservationSpace, \
    TensorObservationSpace, ImageObservationSpace
import numpy as np
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from typing import Union


class GameMap:
    height = 50
    width = 50
    plane = []

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.plane = np.zeros((self.height, self.width, 1), dtype=float)


class Player:
    x = 0
    y = 0

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y


class FacEnv:
    target = None
    wrongs = None
    observation = None
    map_size = 50

    def __init__(self):
        self.wrongs = 0
        # self.target = {'x': 2, 'y': 2}
        self.target = {'x': np.random.randint(1, self.map_size - 2), 'y': np.random.randint(1, self.map_size - 2)}
        print("x: " + str(self.target['x']) + ", y: " + str(self.target['y']))
        self.observation = np.zeros((self.map_size, self.map_size, 1), dtype=float)
        # setting target
        self.observation[self.target['x']][self.target['y']][0] = 1
        self.observation[int(self.map_size / 2)][int(self.map_size / 2)][0] = -1


class TestCls(Environment):
    env = None
    out_of_map_reward = 0
    actions = {}

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters,
                         **kwargs)
        self.env = FacEnv()
        self.movement_reward = None
        self.state_space = StateSpace({
            "observation": PlanarMapsObservationSpace(shape=np.array([self.env.map_size, self.env.map_size, 1]), low=-1,
                                                      high=2)
        })
        # self.goal_space = VectorObservationSpace(shape=4)
        # self.state_space = PlanarMapsObservationSpace(shape=np.array([84, 84, 2]), low=0, high=1)
        self.action_space = DiscreteActionSpace(num_actions=9,
                                                descriptions={"0": "up", "1": "down", "2": "left", "3": "right"})

    def _take_action(self, action_idx: ActionType) -> None:
        self.env.observation[self.env.target['x']][self.env.target['y']][0] = 0
        if int(action_idx) in self.actions:
            self.actions[int(action_idx)] += 1
        else:
            self.actions[int(action_idx)] = 1
        if action_idx == 0:
            self.env.target['y'] += 1
        elif action_idx == 1:
            self.env.target['y'] -= 1
        elif action_idx == 2:
            self.env.target['x'] -= 1
        elif action_idx == 3:
            self.env.target['x'] += 1
        elif action_idx == 4:
            self.env.target['y'] += 1
            self.env.target['x'] += 1
        elif action_idx == 5:
            self.env.target['y'] -= 1
            self.env.target['x'] += 1
        elif action_idx == 6:
            self.env.target['y'] += 1
            self.env.target['x'] -= 1
        elif action_idx == 7:
            self.env.target['y'] -= 1
            self.env.target['x'] -= 1

        if self.env.target['x'] == 0:
            self.env.target['x'] = 1
        if self.env.target['y'] == 0:
            self.env.target['y'] = 1
        if self.env.target['x'] == self.env.map_size - 1:
            self.env.target['x'] = self.env.map_size - 2
        if self.env.target['y'] == self.env.map_size - 1:
            self.env.target['y'] = self.env.map_size - 2

        self.env.observation[int(self.env.map_size / 2)][int(self.env.map_size / 2)][0] = -1
        self.env.observation[self.env.target['x']][self.env.target['y']][0] = 1
        if action_idx == 8:
            # print("MINER: x - " + str(self.env.target['x']) + ", y - " + str(self.env.target['y']))
            if self.env.target['x'] == int(self.env.map_size / 2) and self.env.target['y'] == int(self.env.map_size / 2):
                self.env.observation[self.env.target['x']][self.env.target['y']][0] = 2
        # print("Moved to: ", self.env.target['x'], ", ", self.env.target['y'])

    def _update_state(self) -> None:
        if self.env.observation[int(self.env.map_size/2)][int(self.env.map_size/2)][0] == 2:
            self.finish_reward = 10000
        else:
            self.finish_reward = 0
        self.movement_reward = -(np.abs(int(self.env.map_size/2) - self.env.target['x']) +
                                 np.abs(int(self.env.map_size/2) - self.env.target['y']))
        self.done = (#(self.env.target['x'] == int(self.env.map_size/2) and
                     #self.env.target['y'] == int(self.env.map_size/2)) or
                     self.env.observation[int(self.env.map_size/2)][int(self.env.map_size/2)][0] == 2 or
                     self.current_episode_steps_counter > 200) #or
                     #self.env.target['x'] <= 0 or
                     #self.env.target['y'] <= 0 or
                     #self.env.target['x'] >= self.env.map_size-1 or
                     #self.env.target['y'] >= self.env.map_size-1)
        if self.env.target['x'] == int(self.env.map_size/2) and self.env.target['y'] == int(self.env.map_size/2):
            print("MIDDLE")
        self.state = {"observation": self.env.observation}
        new_reward = self.out_of_map_reward + self.movement_reward + self.finish_reward
        if self.reward < new_reward:
            self.env.wrongs += 1
        self.reward = new_reward
        self.get_action_from_user()
        self.out_of_map_reward = self.movement_reward = self.finish_reward = 0

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        if self.current_episode_steps_counter != 0:
            if self.env.observation[int(self.env.map_size/2)][int(self.env.map_size/2)][0] == 2:
                print(
                    "TARGET after " + str(self.env.wrongs / self.current_episode_steps_counter * 100) + "% wrong steps")
            else:
                print("LOST after " + str(self.env.wrongs / self.current_episode_steps_counter * 100) + "% wrong steps")
        self.env.target = {'x': np.random.randint(1, self.env.map_size-2), 'y': np.random.randint(1, self.env.map_size-2)}
        # self.env.target = {'x': 34, 'y': 12}
        self.env.observation = np.zeros((self.env.map_size, self.env.map_size, 1), dtype=float)
        self.env.observation[self.env.target['x']][self.env.target['y']][0] = 1
        print("Wrongs: ", self.env.wrongs)
        for key in self.actions:
            print(str(key) + ": " + str(self.actions[key]))
        self.actions = {}
        print("x: " + str(self.env.target['x']) + ", y: " + str(self.env.target['y']))
        self.env.wrongs = 0
        self.env.observation[int(self.env.map_size/2)][int(self.env.map_size/2)][0] = -1

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
