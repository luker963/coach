import copy
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
    BELTS = 1


class Miner:
    coordinations = None
    x = 0
    y = 0

    def __init__(self, x, y, orientation):
        self.orientation = orientation
        self.x = x
        self.y = y


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
    channels = len(Channels)
    observation_plane = []
    targets = []
    belts = []

    def __init__(self, height: int, width: int, spawns: int):
        self.targets = []
        self.belts = []
        self.height = height
        self.width = width
        self.observation_plane = np.zeros((self.height, self.width, self.channels), dtype=int)
        for _ in range(spawns):
            while True:
                x = np.random.randint(1, self.height - 2)
                y = np.random.randint(1, self.height - 2)
                target = None
                for target in self.targets:
                    if target.x == x and target.y == y:
                        continue
                if target and target.x == x and target.y == y:
                    continue
                break
            print("spawn at x:" + str(x) + ", y:" + str(y))
            self.observation_plane[x][y][Channels.TARGETS] = 1
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
        self.game_map = GameMap(self.map_size, self.map_size, 2)


class TestCls(Environment):
    out_of_map_reward = 0
    actions = {}

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters,
                         **kwargs)
        self.env = FacEnv()
        self.chest_reward = 0
        self.miner_reward = 0
        self.build_reward = 0
        self.movement_reward = 0
        self.finish_reward = 0
        self.target_with_belt = None
        self.old_distance = 0
        self.new_distance = 0
        self.old_belt_target_distance = 0
        self.new_belt_target_distance = 0
        self.state_space = StateSpace({
            "observation": TensorObservationSpace(shape=np.array([self.env.map_size, self.env.map_size, 2]), low=0,
                                                  high=1)
        })
        self.action_space = DiscreteActionSpace(num_actions=9,
                                                descriptions={})
        self.movements = Movements(self.action_space)

    def is_place_buildable(self):
        if self.env.game_map.observation_plane[int(self.env.map_size / 2)][int(self.env.map_size / 2)][Channels.BELTS] == 0:
            return True
        else:
            return False

    def fill_observations(self):
        self.env.game_map.observation_plane = np.zeros(
            (self.env.map_size, self.env.map_size, self.env.game_map.channels), dtype=int)
        for target in self.env.game_map.targets:
            if 0 <= target.x <= self.env.map_size - 1 and 0 <= target.y <= self.env.map_size - 1:
                self.env.game_map.observation_plane[target.x][target.y][Channels.TARGETS] = 1
        for belts in self.env.game_map.belts:
            if 0 <= belts.x <= self.env.map_size - 1 and 0 <= belts.y <= self.env.map_size - 1:
                self.env.game_map.observation_plane[belts.x][belts.y][Channels.BELTS] = 1

    def compute_reward(self, action_idx):
        if not self.target_with_belt:
            if self.old_distance >= self.new_distance:
                self.movement_reward = self.old_distance - self.new_distance
            else:
                self.movement_reward = -1
            new_reward = self.movement_reward + self.finish_reward + self.chest_reward + self.miner_reward + self.build_reward - 1
        else:
            if self.env.game_map.observation_plane[int(self.env.map_size / 2)][int(self.env.map_size / 2)][Channels.TARGETS] == 1:
                self.build_reward = -1
            elif action_idx == 8:
                self.get_belt_reward()
            else:
                self.movement_reward = -1
            new_reward = self.movement_reward + self.finish_reward + self.chest_reward + self.miner_reward + self.build_reward - 1

        if self.reward >= new_reward:
            self.movements.wrongs += 1
        self.reward = new_reward
        self.movement_reward = self.finish_reward = self.chest_reward = self.miner_reward = self.build_reward = 0

    def get_belt_reward(self):
        desired_target = [target for target in self.env.game_map.targets if target.x != self.target_with_belt.x or target.y != self.target_with_belt.y][0]
        belts_copy = copy.deepcopy(self.env.game_map.belts)
        start_belt = [belt for belt in belts_copy if belt.x == self.target_with_belt.x and belt.y == self.target_with_belt.y][0]
        belts_copy.remove(start_belt)
        remaining_belts = []
        visited_belts = [start_belt]
        self.new_belt_target_distance = self.dist(desired_target, start_belt)
        while True:
            for belt in belts_copy:
                if (belt.x == start_belt.x and belt.y + 1 == start_belt.y) or (belt.x == start_belt.x and belt.y - 1 == start_belt.y) or (belt.x + 1 == start_belt.x and belt.y == start_belt.y) or (belt.x - 1 == start_belt.x and belt.y == start_belt.y):
                    if belt not in visited_belts:
                        if belt not in remaining_belts:
                            remaining_belts.append(belt)
            if len(remaining_belts) == 0:
                self.done = False
                if self.old_belt_target_distance == 0:
                    self.old_belt_target_distance = self.new_belt_target_distance
                elif self.old_belt_target_distance > self.new_belt_target_distance:
                    self.build_reward = 100
                else:
                    self.build_reward = -1
                return
            start_belt = remaining_belts.pop(0)
            self.new_belt_target_distance = min(self.dist(desired_target, start_belt), self.new_belt_target_distance)
            belts_copy.remove(start_belt)
            if start_belt.x == desired_target.x and start_belt.y == desired_target.y:
                self.done = True
                self.build_reward = 100
                return

    def next_belt(self, belt, obs_copy):
        num_of_next_belts = 0
        target = [target for target in self.env.game_map.targets if
                  target.x != self.target_with_belt.x or target.y != self.target_with_belt.y][0]
        if belt.x == target.x and belt.y == target.y:
            return True
        if obs_copy[belt.x][belt.y + 1][Channels.BELTS] == 1:
            num_of_next_belts += 1
        if obs_copy[belt.x][belt.y - 1][Channels.BELTS] == 1:
            num_of_next_belts += 1
        if obs_copy[belt.x + 1][belt.y][Channels.BELTS] == 1:
            num_of_next_belts += 1
        if obs_copy[belt.x - 1][belt.y][Channels.BELTS] == 1:
            num_of_next_belts += 1
        obs_copy[belt.x][belt.y][Channels.BELTS] = 0
        if obs_copy[belt.x][belt.y + 1][Channels.BELTS] == 1:
            self.next_belt(Coordinate(belt.x, belt.y + 1), obs_copy)
        if obs_copy[belt.x][belt.y - 1][Channels.BELTS] == 1:
            self.next_belt(Coordinate(belt.x, belt.y - 1), obs_copy)
        if obs_copy[belt.x + 1][belt.y][Channels.BELTS] == 1:
            self.next_belt(Coordinate(belt.x + 1, belt.y), obs_copy)
        if obs_copy[belt.x - 1][belt.y][Channels.BELTS] == 1:
            self.next_belt(Coordinate(belt.x - 1, belt.y), obs_copy)
        return False

    def _take_action(self, action_idx: ActionType) -> None:
        self.chest_reward = 0
        self.miner_reward = 0
        self.build_reward = 0
        self.movement_reward = 0
        self.finish_reward = 0
        self.dist = lambda a, b: np.abs(a.x - b.x) + np.abs(a.y - b.y)
        closest_target = min(self.env.game_map.targets,
                             key=lambda co: self.dist(co, Coordinate(int(self.env.map_size / 2),
                                                                     int(self.env.map_size / 2))))
        self.old_distance = np.abs(int(self.env.map_size / 2) - closest_target.x) + np.abs(
            int(self.env.map_size / 2) - closest_target.y)
        self.movements.actions_counter[action_idx] += 1
        if action_idx == 8:
            if self.is_place_buildable():
                new_belt = Coordinate(int(self.env.map_size / 2), int(self.env.map_size / 2))
                self.env.game_map.belts.append(new_belt)
                if self.env.game_map.observation_plane[int(self.env.map_size / 2)][int(self.env.map_size / 2)][Channels.TARGETS] == 1:
                    self.build_reward = 100
                    print("belt on target")
                    self.target_with_belt = [target for target in self.env.game_map.targets if
                                             target.x == target.y == int(self.env.map_size / 2)][0]
                else:
                    self.build_reward = -1
            else:
                self.build_reward = -1
        else:
            for entity in self.env.game_map.targets + self.env.game_map.belts:
                if action_idx == 0:
                    entity.y += 1
                elif action_idx == 1:
                    entity.y -= 1
                elif action_idx == 2:
                    entity.x -= 1
                elif action_idx == 3:
                    entity.x += 1
                elif action_idx == 4:
                    entity.x += 1
                    entity.y += 1
                elif action_idx == 5:
                    entity.x += 1
                    entity.y -= 1
                elif action_idx == 6:
                    entity.x -= 1
                    entity.y += 1
                elif action_idx == 7:
                    entity.x -= 1
                    entity.y -= 1
        self.fill_observations()
        closest_target = min(self.env.game_map.targets, key=lambda co: self.dist(co,
                                                                                 Coordinate(int(self.env.map_size / 2),
                                                                                            int(
                                                                                                self.env.map_size / 2))))
        self.new_distance = np.abs(
            int(self.env.map_size / 2) - closest_target.x) + np.abs(int(self.env.map_size / 2) - closest_target.y)
        self.compute_reward(action_idx)

    def _update_state(self) -> None:
            # self.done = self.next_belt(self.env.game_map.belts[0], copy.deepcopy(self.env.game_map.observation_plane))
        if self.done:
            print("TARGET")
        self.done = (self.done or
                     self.current_episode_steps_counter > 200)
        self.state = {"observation": self.env.game_map.observation_plane}

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        self.target_with_belt = None
        print("Wrongs: ", self.movements.wrongs)
        for index, value in enumerate(self.movements.actions_counter):
            print(str(index) + ": " + str(value))
        self.env = FacEnv()
        self.movements = Movements(self.action_space)

    def get_rendered_image(self):
        image_map = np.copy(self.env.game_map.observation_plane[:, :, 0]).astype(float)
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
