from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import ActionType
from rl_coach.environments.environment import Environment, LevelSelection
from rl_coach.spaces import PlanarMapsObservationSpace, DiscreteActionSpace, StateSpace, VectorObservationSpace, \
    TensorObservationSpace, ImageObservationSpace
import numpy as np
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from typing import Union


class Movements:
    wrongs = 0
    actions_counter = None

    def __init__(self, action_space):
        self.actions_counter = np.zeros(len(action_space.actions))


class GameMap:
    height = 50
    width = 50
    channels = 2
    plane = []
    target_x = 0
    target_y = 0

    def __init__(self, height: int, width: int, target_x: int, target_y: int):
        self.height = height
        self.width = width
        self.plane = np.zeros((self.height, self.width, self.channels), dtype=int)
        self.target_x = target_x
        self.target_y = target_y
        self.plane[target_x][target_y][0] = 1


class Player:
    x = 0
    y = 0

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y


class FacEnv:
    game_map = None
    observation = None
    map_size = 50
    # player = None

    def __init__(self):
        self.game_map = GameMap(self.map_size, self.map_size, np.random.randint(1, self.map_size - 2), np.random.randint(1, self.map_size - 2))
        # self.player = Player(int(self.map_size / 2), int(self.map_size / 2))
        print("Destination x: " + str(self.game_map.target_x) + ", y: " + str(self.game_map.target_y))
        self.observation = self.game_map.plane
        # self.observation[self.player.x][self.player.y][0] = -100


class TestCls(Environment):
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
            "observation": PlanarMapsObservationSpace(shape=np.array([self.env.map_size, self.env.map_size, 2]), low=0,
                                                      high=1)
        })
        self.action_space = DiscreteActionSpace(num_actions=9,
                                                descriptions={"0": "up", "1": "down", "2": "left", "3": "right"})
        self.movements = Movements(self.action_space)

    def _take_action(self, action_idx: ActionType) -> None:
        self.env.observation[self.env.game_map.target_x][self.env.game_map.target_y][0] = 0
        self.movements.actions_counter[action_idx] += 1
        if action_idx == 0:
            self.env.game_map.target_y += 1
        elif action_idx == 1:
            self.env.game_map.target_y -= 1
        elif action_idx == 2:
            self.env.game_map.target_x -= 1
        elif action_idx == 3:
            self.env.game_map.target_x += 1
        elif action_idx == 4:
            self.env.game_map.target_y += 1
            self.env.game_map.target_x += 1
        elif action_idx == 5:
            self.env.game_map.target_y -= 1
            self.env.game_map.target_x += 1
        elif action_idx == 6:
            self.env.game_map.target_y += 1
            self.env.game_map.target_x -= 1
        elif action_idx == 7:
            self.env.game_map.target_y -= 1
            self.env.game_map.target_x -= 1

        if self.env.game_map.target_x == 0:
            self.env.game_map.target_x = 1
        if self.env.game_map.target_y == 0:
            self.env.game_map.target_y = 1
        if self.env.game_map.target_x == self.env.map_size - 1:
            self.env.game_map.target_x = self.env.map_size - 2
        if self.env.game_map.target_y == self.env.map_size - 1:
            self.env.game_map.target_y = self.env.map_size - 2

        self.env.observation[self.env.game_map.target_x][self.env.game_map.target_y][0] = 1
        if action_idx == 8:
            if self.env.game_map.target_x == int(self.env.map_size/2) and self.env.game_map.target_y == int(self.env.map_size/2):
                self.env.game_map.plane[int(self.env.map_size/2)][int(self.env.map_size/2)][1] = 1

    def _update_state(self) -> None:
        if self.env.game_map.plane[int(self.env.map_size/2)][int(self.env.map_size/2)][1] == 1:
            print(
                "TARGET after " + str(
                    self.movements.wrongs / self.current_episode_steps_counter * 100) + "% wrong steps")
            self.finish_reward = 10000
        else:
            self.finish_reward = 0
        if self.env.game_map.plane[int(self.env.map_size/2)][int(self.env.map_size/2)][0] == 1:
            print("MIDDLE")
        self.movement_reward = -(np.abs(int(self.env.map_size/2) - self.env.game_map.target_x) +
                                 np.abs(int(self.env.map_size/2) - self.env.game_map.target_y))
        self.done = (self.env.game_map.plane[int(self.env.map_size/2)][int(self.env.map_size/2)][1] == 1 or
                     self.current_episode_steps_counter > 200)
        self.state = {"observation": self.env.observation}
        new_reward = self.out_of_map_reward + self.movement_reward + self.finish_reward
        if self.reward >= new_reward:
            self.movements.wrongs += 1
        self.reward = new_reward
        self.out_of_map_reward = self.movement_reward = self.finish_reward = 0
        if self.current_episode_steps_counter > 200:
            print(
                "LOST after " + str(self.movements.wrongs / self.current_episode_steps_counter * 100) + "% wrong steps")

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        print("Wrongs: ", self.movements.wrongs)
        for index, value in enumerate(self.movements.actions_counter):
            print(str(index) + ": " + str(value))
        self.env = FacEnv()
        self.movements = Movements(self.action_space)

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
