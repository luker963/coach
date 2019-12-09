import gc
import logging
import sys
from enum import IntEnum
from typing import Union

import numpy as np

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import ActionType, RunPhase
from rl_coach.environments.environment import Environment, LevelSelection, EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.spaces import StateSpace, DiscreteActionSpace, PlanarMapsObservationSpace, TensorObservationSpace, \
    VectorObservationSpace


class Channels(IntEnum):
    ENTITY = 0
    # ORIENTATION = 1
    CONTENT = 1
    ORE = 2
    PLAYER = 3


class Entities(IntEnum):
    EMPTY = 0
    MINER = 1
    BELT = 2
    INSERTER = 3
    CHEST = 4


class Orientations(IntEnum):
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7

    @classmethod
    def opposite_of(cls, orientation):
        if orientation == cls.UP:
            return cls.DOWN
        if orientation == cls.DOWN:
            return cls.UP
        if orientation == cls.LEFT:
            return cls.RIGHT
        if orientation == cls.RIGHT:
            return cls.LEFT


class Player:
    x = 0
    y = 0

    def __init__(self, x: int = 25, y: int = 25):
        self.x = x
        self.y = y


class FactEnv(Environment):

    def get_observation_from_map(self):
        x_obs_from = int(self.player.x - 2)
        x_obs_to = int(self.player.x + 2)
        y_obs_from = int(self.player.y - 2)
        y_obs_to = int(self.player.y + 2)
        return self.map[x_obs_from:x_obs_to + 1, y_obs_from:y_obs_to + 1]

    def spawn_ore(self):
        ore_coord_x = np.random.randint(10, 40)
        ore_coord_y = np.random.randint(10, 40)
        self.map[ore_coord_x, ore_coord_y, Channels.ORE] = 20
        # for chunk in range(self.ore_chunks):
        #     ore_coord_x = np.random.randint(10, 40)
        #     ore_coord_y = np.random.randint(10, 40)
        #     for x in range(ore_coord_x - self.ore_diameter, ore_coord_x + self.ore_diameter):
        #         for y in range(ore_coord_y - self.ore_diameter, ore_coord_y + self.ore_diameter):
        #             self.map[x, y, Channels.ORE] = 20

    def _move_player(self, action_idx):
        old_player = Player(self.player.x, self.player.y)
        self.map[self.player.x, self.player.y, Channels.PLAYER] = 0
        if action_idx == 0:
            self.player.y += 1
        if action_idx == 1:
            self.player.y -= 1
        if action_idx == 2:
            self.player.x -= 1
        if action_idx == 3:
            self.player.x += 1
        if 5 > self.player.x or self.player.x >= self.map_width - 5 or 5 > self.player.y or self.player.y >= self.map_height - 5:
            self.new_reward += -20
            self.player = old_player
            self.map[self.player.x, self.player.y, Channels.PLAYER] = 1
            self.movements_out += 1
        else:
            self.map[self.player.x, self.player.y, Channels.PLAYER] = 1
            self.new_reward += -10
            self.movements += 1

    def _place_oriented(self, action_idx):
        if 1 > self.player.x or self.player.x > self.map_width - 2 or 1 > self.player.y or self.player.y > self.map_height - 2:  # out of map bound
            pass
        elif self.map[self.player.x, self.player.y, Channels.ENTITY] > 0:  # placing on top of something
            self.new_reward = -100
        elif action_idx < 12:  # MINER
            if self.map[self.player.x, self.player.y, Channels.ORE] == 20:  # if ore is present
                self.map[self.player.x, self.player.y, Channels.ENTITY] = Entities.MINER
                self.miners.append((self.player.x, self.player.y))
                self.new_reward += 2000
                self.logger.info("[PLACEMENT]: Miner placed at x:{} y:{}".format(self.player.x, self.player.y))
            else:
                self.new_reward = -100
                self.logger.debug("[PLACEMENT]: unable to place miner without ore")

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters,
                         **kwargs)

        self.logger = logging.getLogger("RL_logger")
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.movements_out = 0
        self.movements = 0
        self.chests = []
        self.belts = []
        self.inserters = []
        self.miners = []
        self.minimum_reward = -100000
        self.new_reward = 0
        self.map_height = self.map_width = 50
        self.observation_radius = 5
        self.ore_diameter = 4
        self.ore_chunks = 5
        self.player = Player(25, 25)
        self.map = np.zeros((self.map_height, self.map_width, 4))
        self.spawn_ore()

        self.state_space = StateSpace(
            {"observation": PlanarMapsObservationSpace(np.array([50, 50, 4]), low=0, high=20),
             "player": VectorObservationSpace(2)}
             # }
        )
        self.action_space = DiscreteActionSpace(num_actions=5,
                                                descriptions={"0": "up", "1": "down", "2": "left", "3": "right",
                                                              "4": "mine-up"})

    def init_env(self):
        self.movements_out = 0
        self.movements = 0
        self.chests = []
        self.belts = []
        self.inserters = []
        self.miners = []
        self.minimum_reward = -10000
        self.new_reward = 0
        self.map_height = self.map_width = 50
        self.observation_radius = 5
        self.ore_diameter = 2
        self.player = Player(25, 25)
        # self.logger.info("[PLAYER]: Spawned at x: {} y: {}".format(self.player.x, self.player.y))
        self.map = np.zeros((self.map_height, self.map_width, 4))
        self.spawn_ore()

    def _take_action(self, action_idx: ActionType) -> None:
        self.new_reward = 0
        if action_idx < 4:
            self._move_player(action_idx)
        elif action_idx < 20:
            self._place_oriented(action_idx)

    def _update_state(self) -> None:
        self.done = False
        self.reward = self.new_reward
        if self.phase == RunPhase.HEATUP:
            if self.current_episode_steps_counter > 100000:
                self.done = True
        elif self.current_episode_steps_counter > 300:
            self.done = True
            self.logger.info("[TERMINATION]: Exceed maximum steps 100")
            self.logger.info("[TERMINATION]: Total movements: {}, wrongs: {}, miners: {}".format(self.movements, self.movements_out, len(self.miners)))
        elif self.total_reward_in_current_episode < self.minimum_reward:
            self.done = True
            self.logger.info("[TERMINATION]: Exceed minimum reward {}".format(self.minimum_reward))
            self.logger.info("[TERMINATION]: Total movements: {}, wrongs: {}".format(self.movements, self.movements_out))
        elif len(self.miners) > 0:
            #TODO: nechat ho dlhsie pracovat
            self.done = True
            self.reward = 3000
            self.logger.info("[TERMINATION]: Total movements: {}, wrongs: {}".format(self.movements, self.movements_out))

        self.state = {'observation': self.map, 'player': [self.player.x, self.player.x]}

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        self.init_env()

    def get_rendered_image(self) -> np.ndarray:
        image_map = np.copy(self.map[:, :, Channels.ORE])
        image_map[image_map == 0] = 255
        image_map[image_map == 20] = 170
        image_map[self.player.x, self.player.y] = 85
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
        return 'rl_coach.environments.second_test:FactEnv'
