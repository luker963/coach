from enum import IntEnum

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import ActionType
from rl_coach.environments.environment import Environment, LevelSelection
from rl_coach.spaces import PlanarMapsObservationSpace, DiscreteActionSpace, StateSpace, VectorObservationSpace, \
    TensorObservationSpace, ImageObservationSpace
import numpy as np
from rl_coach.environments.environment import EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from typing import Union


class Channels(IntEnum):
    TARGETS = 0
    PLAYER = 0


class Coordinate:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Movements:
    wrongs = 0
    actions_counter = None

    def __init__(self, action_space):
        self.actions_counter = np.zeros(len(action_space.actions))


class GameMap:
    height = 50
    width = 50
    channels = 1
    targets_plane = []
    targets = []

    def __init__(self, height: int, width: int, spawns: int):
        self.targets = []
        self.targets_plane = []
        self.height = height
        self.width = width
        self.targets_plane = np.zeros((self.height, self.width, self.channels), dtype=float)
        # self.targets_plane[int(self.height / 2)][int(self.height / 2)][Channels.PLAYER] = 2
        for _ in range(spawns):
            x = np.random.randint(1, self.height - 2)
            y = np.random.randint(1, self.height - 2)
            print("spawn at x:" + str(x) + ", y:" + str(y))
            self.targets_plane[x][y][Channels.TARGETS] = 1
            self.targets.append(Coordinate(x, y))


class Player:
    x = 0
    y = 0

    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y


class FacEnv:
    game_map = None
    map_size = 50

    def __init__(self):
        self.game_map = GameMap(self.map_size, self.map_size, 1)
        # self.player = Player(int(self.map_size / 2), int(self.map_size / 2))
        # print("Destination x: " + str(self.game_map.target_x) + ", y: " + str(self.game_map.target_y))


class TestCls(Environment):
    out_of_map_reward = 0
    actions = {}

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters,
                         **kwargs)
        self.env = FacEnv()
        self.movement_reward = 0
        self.finish_reward = 0
        self.old_distance = 0
        self.new_distance = 0
        self.state_space = StateSpace({
            "observation": TensorObservationSpace(shape=np.array([self.env.map_size, self.env.map_size, 1]), low=0,
                                                      high=1)
        })
        self.action_space = DiscreteActionSpace(num_actions=9,
                                                descriptions={"0": "up", "1": "down", "2": "left", "3": "right"})
        self.movements = Movements(self.action_space)

    def _take_action(self, action_idx: ActionType) -> None:
        dist = lambda a, b: (a.x - b.x) ** 2 + (a.y - b.y) ** 2
        closest_target = min(self.env.game_map.targets, key=lambda co: dist(co, Coordinate(int(self.env.map_size / 2),
                                                                                           int(self.env.map_size / 2))))
        self.old_distance = np.abs(int(self.env.map_size/2) - closest_target.x) + np.abs(int(self.env.map_size/2) - closest_target.y)
        self.movements.actions_counter[action_idx] += 1
        if action_idx == 8:
            if self.env.game_map.targets_plane[int(self.env.map_size/2)][int(self.env.map_size/2)][0] == 1:
                self.done = True
                self.finish_reward = 10000
            else:
                self.finish_reward = -1
        else:
            for target in self.env.game_map.targets:
                if 0 <= target.x <= self.env.map_size - 1 and 0 <= target.y <= self.env.map_size - 1:
                    self.env.game_map.targets_plane[target.x][target.y][0] = 0
                if action_idx == 0:
                    target.y += 1
                elif action_idx == 1:
                    target.y -= 1
                elif action_idx == 2:
                    target.x -= 1
                elif action_idx == 3:
                    target.x += 1
                elif action_idx == 4:
                    target.x += 1
                    target.y += 1
                elif action_idx == 5:
                    target.x += 1
                    target.y -= 1
                elif action_idx == 6:
                    target.x -= 1
                    target.y += 1
                elif action_idx == 7:
                    target.x -= 1
                    target.y -= 1
                if 0 <= target.x <= self.env.map_size - 1 and 0 <= target.y <= self.env.map_size - 1:
                    self.env.game_map.targets_plane[target.x][target.y][0] = 1
            closest_target = min(self.env.game_map.targets, key=lambda co: dist(co,
                                                                                Coordinate(int(self.env.map_size / 2),
                                                                                           int(self.env.map_size / 2))))
            self.new_distance = np.abs(
                int(self.env.map_size / 2) - closest_target.x) + np.abs(int(self.env.map_size / 2) - closest_target.y)

    def _update_state(self) -> None:
        # dist = lambda a, b: (a.x - b.x) ** 2 + (a.y - b.y) ** 2
        # closest_tartet = min(self.env.game_map.targets, key=lambda co: dist(co, Coordinate(int(self.env.map_size/2), int(self.env.map_size/2))))
        # self.new_distance = np.abs(int(self.env.map_size / 2) - closest_tartet.x + np.abs(int(self.env.map_size / 2) - closest_tartet.y))
        # closest_target = self.env.game_map.targets[0]
        # if self.env.game_map.targets_plane[int(self.env.map_size/2)][int(self.env.map_size/2)][0] == 1:
            # print("MIDDLE")
            # self.done = True
        if self.done:
            print("TARGET")
            self.finish_reward = 10000
        else:
            self.finish_reward = 0

        # self.movement_reward = -(np.abs(int(self.env.map_size/2) - closest_target.x) +
        #                          np.abs(int(self.env.map_size/2) - closest_target.y))
        self.movement_reward = 0
        if self.old_distance != self.new_distance:
            self.movement_reward = self.old_distance - self.new_distance
        else:
            self.movement_reward = -1
        self.done = (self.done or
                     self.current_episode_steps_counter > 100)
        self.state = {"observation": self.env.game_map.targets_plane}
        new_reward = self.out_of_map_reward + self.movement_reward + self.finish_reward
        if self.reward >= new_reward:
            self.movements.wrongs += 1
        self.reward = new_reward
        self.out_of_map_reward = self.movement_reward = self.finish_reward = 0
        # if self.current_episode_steps_counter > 100:
        #     print("LOST")

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        print("Wrongs: ", self.movements.wrongs)
        for index, value in enumerate(self.movements.actions_counter):
            print(str(index) + ": " + str(value))
        self.env = FacEnv()
        self.movements = Movements(self.action_space)

    def get_rendered_image(self):
        image_map = np.copy(self.env.game_map.targets_plane[:, :, 0]).astype(float)
        image_map[image_map == 0] = 255
        image_map[image_map == 1] = 170
        return image_map


class ControlSuiteEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()

    @property
    def path(self):
        return 'rl_coach.environments.first_test:TestCls'
