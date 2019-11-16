import logging
import sys
from enum import IntEnum
from typing import Union
import numpy as np
import gc

from rl_coach.base_parameters import VisualizationParameters
from rl_coach.core_types import ActionType
from rl_coach.environments.environment import Environment, LevelSelection, EnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
from rl_coach.spaces import StateSpace, DiscreteActionSpace, PlanarMapsObservationSpace


class Channels(IntEnum):
    ENTITY = 0
    # ORIENTATION = 1
    ORE = 1
    CONTENT = 2
    PLAYER = 3
    UP = 4
    DOWN = 5
    LEFT = 6
    RIGHT = 7


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
    map_height = map_width = 50
    map = None
    player = None
    miners = None
    inserters = None
    belts = None
    chests = None
    new_reward = None
    logger = logging.getLogger("RL_logger")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    minimum_reward = None

    def get_map_oriented_increment(self, x, y, channel, orientation):
        if orientation == Orientations.UP:
            return self.map[x][y + 1][channel]
        if orientation == Orientations.DOWN:
            return self.map[x][y - 1][channel]
        if orientation == Orientations.LEFT:
            return self.map[x - 1][y][channel]
        if orientation == Orientations.RIGHT:
            return self.map[x + 1][y][channel]

    def set_map_oriented_increment(self, x, y, channel, orientation, value):
        if orientation == Orientations.UP:
            self.map[x][y + 1][channel] = value
        if orientation == Orientations.DOWN:
            self.map[x][y - 1][channel] = value
        if orientation == Orientations.LEFT:
            self.map[x - 1][y][channel] = value
        if orientation == Orientations.RIGHT:
            self.map[x + 1][y][channel] = value

    def spawn_ore(self):
        ore_coord_x = np.random.randint(10, 40)
        ore_coord_y = np.random.randint(10, 40)
        for x in range(ore_coord_x - 5, ore_coord_x + 5):
            dist = 5 - np.abs(x - ore_coord_x)
            for y in range(ore_coord_y - dist, ore_coord_y + dist):
                self.map[x][y][Channels.ORE] = 1

    def update_map(self):
        self.logger.debug("[MAP UPDATE]: Starting map update")
        self.logger.debug("[MAP UPDATE]: Miners count: {}".format(len(self.miners)))
        self.logger.debug("[MAP UPDATE]: Belts count: {}".format(len(self.belts)))
        self.logger.debug("[MAP UPDATE]: Inserters count: {}".format(len(self.inserters)))
        self.logger.debug("[MAP UPDATE]: Chests count: {}".format(len(self.chests)))

        for miner in self.miners:
            if self.map[miner[0]][miner[1]][Channels.UP] and not \
                    self.get_map_oriented_increment(miner[0], miner[1], Channels.CONTENT, Orientations.UP):  # self.map[miner[0]][miner[1] + 1][Channels.CONTENT] == 0:
                self.map[miner[0]][miner[1] + 1][Channels.CONTENT] += 1
                self.logger.info("[MAP UPDATE]: Miner at {},{} produced ore UP".format(miner[0], miner[1]))
            if self.map[miner[0]][miner[1]][Channels.DOWN] and not \
                    self.get_map_oriented_increment(miner[0], miner[1], Channels.CONTENT, Orientations.DOWN):
                self.map[miner[0]][miner[1] - 1][Channels.CONTENT] += 1
                self.logger.info("[MAP UPDATE]: Miner at {},{} produced ore DOWN".format(miner[0], miner[1]))
            if self.map[miner[0]][miner[1]][Channels.LEFT] and not \
                    self.get_map_oriented_increment(miner[0], miner[1], Channels.CONTENT, Orientations.LEFT):
                self.map[miner[0] - 1][miner[1]][Channels.CONTENT] += 1
                self.logger.info("[MAP UPDATE]: Miner at {},{} produced ore LEFT".format(miner[0], miner[1]))
            if self.map[miner[0]][miner[1]][Channels.RIGHT] and not \
                    self.get_map_oriented_increment(miner[0], miner[1], Channels.CONTENT, Orientations.RIGHT):
                self.map[miner[0] + 1][miner[1]][Channels.CONTENT] += 1
                self.logger.info("[MAP UPDATE]: Miner at {},{} produced ore RIGHT".format(miner[0], miner[1]))
        for inserter in self.inserters:
            if self.map[inserter[0]][inserter[1]][Channels.UP]:
                if self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.DOWN) and not self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.UP):
                    self.map[inserter[0]][inserter[1] - 1][Channels.CONTENT] -= 1
                    self.map[inserter[0]][inserter[1] + 1][Channels.CONTENT] += 1
                    self.logger.info("[MAP UPDATE]: Inserter at {},{} moved ore UP".format(inserter[0], inserter[1]))
            if self.map[inserter[0]][inserter[1]][Channels.DOWN]:
                if self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.UP) and not self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.DOWN):
                    self.map[inserter[0]][inserter[1] + 1][Channels.CONTENT] -= 1
                    self.map[inserter[0]][inserter[1] - 1][Channels.CONTENT] += 1
                    self.logger.info("[MAP UPDATE]: Inserter at {},{} moved ore DOWN".format(inserter[0], inserter[1]))
            if self.map[inserter[0]][inserter[1]][Channels.LEFT]:
                if self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.RIGHT) and not self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.LEFT):
                    self.map[inserter[0] + 1][inserter[1]][Channels.CONTENT] -= 1
                    self.map[inserter[0] - 1][inserter[1]][Channels.CONTENT] += 1
                    self.logger.info("[MAP UPDATE]: Inserter at {},{} moved ore LEFT".format(inserter[0], inserter[1]))
            if self.map[inserter[0]][inserter[1]][Channels.RIGHT]:
                if self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.LEFT) and not self.get_map_oriented_increment(inserter[0], inserter[1], Channels.CONTENT, Orientations.RIGHT):
                    self.map[inserter[0] - 1][inserter[1]][Channels.CONTENT] -= 1
                    self.map[inserter[0] + 1][inserter[1]][Channels.CONTENT] += 1
                    self.logger.info("[MAP UPDATE]: Inserter at {},{} moved ore RIGHT".format(inserter[0], inserter[1]))
        for belt in self.belts:
            if self.map[belt[0]][belt[1]][Channels.UP]:
                if self.map[belt[0]][belt[1]][Channels.CONTENT] == 0 and self.get_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.DOWN) and self.get_map_oriented_increment(belt[0], belt[1], Channels.ENTITY, Orientations.DOWN) == Entities.BELT:
                # if self.map[belt[0]][belt[1]][Channels.CONTENT] > 0 and not self.map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.UP):
                    self.set_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.DOWN, 0)
                    self.map[belt[0]][belt[1]][Channels.CONTENT] = 1
                    self.logger.info("[MAP UPDATE]: Belt at {},{} pulled ore UP".format(belt[0], belt[1]))
            if self.map[belt[0]][belt[1]][Channels.DOWN]:
                if self.map[belt[0]][belt[1]][Channels.CONTENT] == 0 and self.get_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.UP) and self.get_map_oriented_increment(belt[0], belt[1], Channels.ENTITY, Orientations.UP) == Entities.BELT:
                    self.set_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.UP, 0)
                    self.map[belt[0]][belt[1]][Channels.CONTENT] = 1
                    self.logger.info("[MAP UPDATE]: Belt at {},{} pulled ore DOWN".format(belt[0], belt[1]))
            if self.map[belt[0]][belt[1]][Channels.LEFT]:
                if self.map[belt[0]][belt[1]][Channels.CONTENT] == 0 and self.get_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.RIGHT) and self.get_map_oriented_increment(belt[0], belt[1], Channels.ENTITY, Orientations.RIGHT) == Entities.BELT:
                    self.set_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.RIGHT, 0)
                    self.map[belt[0]][belt[1]][Channels.CONTENT] = 1
                    self.logger.info("[MAP UPDATE]: Belt at {},{} pulled ore LEFT".format(belt[0], belt[1]))
            if self.map[belt[0]][belt[1]][Channels.RIGHT]:
                if self.map[belt[0]][belt[1]][Channels.CONTENT] == 0 and self.get_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.LEFT) and self.get_map_oriented_increment(belt[0], belt[1], Channels.ENTITY, Orientations.LEFT) == Entities.BELT:
                    self.set_map_oriented_increment(belt[0], belt[1], Channels.CONTENT, Orientations.LEFT, 0)
                    self.map[belt[0]][belt[1]][Channels.CONTENT] = 1
                    self.logger.info("[MAP UPDATE]: Belt at {},{} pulled ore RIGHT".format(belt[0], belt[1]))

    # Map definition:
    #   [x][y][0] == 1 is player location
    #   [x][y][0] > 1 is building placed
    #   [x][y][1] is building orientation (applicable only if [x][y][1] > 1)
    #       0 - up
    #       1 - down
    #       2 - left
    #       3 - right
    #   [x][y][2] > 1 is ore present
    #   [x][y][3] content of entity

    def init_env(self):
        self.map = np.zeros((self.map_height, self.map_width, len(Channels)))
        self.spawn_ore()
        self.player = Player()
        self.miners = []
        self.inserters = []
        self.belts = []
        self.chests = []
        self.new_reward = 0
        self.minimum_reward = -100000

    def _move_player(self, action_idx):
        old_player = Player(self.player.x, self.player.y)
        self.map[self.player.x][self.player.y][Channels.PLAYER] = 0
        if action_idx == 0:
            self.player.y += 1
        if action_idx == 1:
            self.player.y -= 1
        if action_idx == 2:
            self.player.x -= 1
        if action_idx == 3:
            self.player.x += 1
        if action_idx == 4:
            self.player.y += 1
            self.player.x -= 1
        if action_idx == 5:
            self.player.y += 1
            self.player.x += 1
        if action_idx == 6:
            self.player.x -= 1
            self.player.y -= 1
        if action_idx == 7:
            self.player.x += 1
            self.player.y -= 1
        if 0 > self.player.x or self.player.x >= self.map_width or 0 > self.player.y or self.player.y >= self.map_height:
            # self.new_reward += -100
            self.player = old_player
            self.map[self.player.x][self.player.y][Channels.PLAYER] = 1
        else:
            self.map[self.player.x][self.player.y][Channels.PLAYER] = 1
            # self.new_reward += -50

    def _place_oriented(self, action_idx):
        if 1 > self.player.x or self.player.x > self.map_width - 2 or 1 > self.player.y or self.player.y > self.map_height - 2:  # out of map bound
            pass
            # self.new_reward = -500
        elif self.map[self.player.x][self.player.y][Channels.ENTITY] > 0:  # placing on top of something
            pass
            # self.new_reward = -500
            # print("PLACEMENT: unable to place, there is already {}".format(self.map[self.player.x][self.player.y][0]))
        elif action_idx < 12:  # MINER
            if self.map[self.player.x][self.player.y][Channels.ORE] == 1:  # if ore is present
                self.map[self.player.x][self.player.y][Channels.ENTITY] = Entities.MINER
                if action_idx == 8:
                    self.map[self.player.x][self.player.y][Channels.UP] = 1
                    if self.map[self.player.x][self.player.y + 1][Channels.ENTITY] == Entities.BELT and not \
                            self.map[self.player.x][self.player.y + 1][Channels.DOWN]:
                        self.new_reward += 2000
                        self.logger.info("[PLACEMENT]: Miner placed next to the belt")
                elif action_idx == 9:
                    self.map[self.player.x][self.player.y][Channels.DOWN] = 1
                    if self.map[self.player.x][self.player.y - 1][Channels.ENTITY] == Entities.BELT and not \
                            self.map[self.player.x][self.player.y - 1][Channels.UP]:
                        self.new_reward += 2000
                        self.logger.info("[PLACEMENT]: Miner placed next to the belt")
                elif action_idx == 10:
                    self.map[self.player.x][self.player.y][Channels.LEFT] = 1
                    if self.map[self.player.x - 1][self.player.y][Channels.ENTITY] == Entities.BELT and not \
                            self.map[self.player.x - 1][self.player.y][Channels.RIGHT]:
                        self.new_reward += 2000
                        self.logger.info("[PLACEMENT]: Miner placed next to the belt")
                elif action_idx == 11:
                    self.map[self.player.x][self.player.y][Channels.RIGHT] = 1
                    if self.map[self.player.x + 1][self.player.y][Channels.ENTITY] == Entities.BELT and not \
                            self.map[self.player.x + 1][self.player.y][Channels.LEFT]:
                        self.new_reward += 2000
                        self.logger.info("[PLACEMENT]: Miner placed next to the belt")
                self.miners.append((self.player.x, self.player.y))
                self.new_reward += 100
                # self.new_reward -= 50*len(self.miners)
                self.logger.info("[PLACEMENT]: Miner placed at x:{} y:{}".format(self.player.x, self.player.y))
                # self.minimum_reward = -500000
            else:
                # self.new_reward = -100
                self.logger.debug("[PLACEMENT]: unable to place miner without ore")
        elif action_idx < 16:  # BELT
            self.map[self.player.x][self.player.y][Channels.ENTITY] = Entities.BELT
            if action_idx == 12:
                self.map[self.player.x][self.player.y][Channels.UP] = 1
                if self.map[self.player.x][self.player.y + 1][Channels.ENTITY] == Entities.BELT and not \
                        self.map[self.player.x][self.player.y + 1][Channels.DOWN]:
                    self.new_reward += 100
            elif action_idx == 13:
                self.map[self.player.x][self.player.y][Channels.DOWN] = 1
                if self.map[self.player.x][self.player.y - 1][Channels.ENTITY] == Entities.BELT and not \
                        self.map[self.player.x][self.player.y - 1][Channels.UP]:
                    self.new_reward += 100
            elif action_idx == 14:
                self.map[self.player.x][self.player.y][Channels.LEFT] = 1
                if self.map[self.player.x - 1][self.player.y][Channels.ENTITY] == Entities.BELT and not \
                        self.map[self.player.x - 1][self.player.y][Channels.RIGHT]:
                    self.new_reward += 100
            elif action_idx == 15:
                self.map[self.player.x][self.player.y][Channels.RIGHT] = 1
                if self.map[self.player.x + 1][self.player.y][Channels.ENTITY] == Entities.BELT and not \
                        self.map[self.player.x + 1][self.player.y][Channels.LEFT]:
                    self.new_reward += 100
            self.belts.append((self.player.x, self.player.y))
            for orientation in Orientations:
                # ak je v okoli belt
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   orientation) == Entities.BELT:
                    # ak ten belt smeruje na poziciu originu
                    if self.get_map_oriented_increment(self.player.x, self.player.y, Orientations.opposite_of(orientation),
                                                       orientation):
                        # ak nema belt v origine orientaciu kolidujucu s beltom v okoli
                        if not self.map[self.player.x][self.player.y][orientation]:
                            self.new_reward += 500
                            self.logger.info("[PLACEMENT]: Belt placed next to the belt")
                        else:  # ak ma belt v origine orientaciu kolidujucu s beltom v okoli
                            self.new_reward += -500
                            self.logger.info("[PLACEMENT]: Belt placed next to the belt but WRONG")
                # ak je v okoli miner
                elif self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                     orientation) == Entities.MINER:
                    # ak ten miner smeruje na poziciu originu
                    if self.get_map_oriented_increment(self.player.x, self.player.y, Orientations.opposite_of(orientation),
                                                       orientation):
                        # ak nema belt v origine orientaciu kolidujucu s minerom v okoli
                        if not self.map[self.player.x][self.player.y][orientation]:
                            self.new_reward += 2000
                            self.logger.info("[PLACEMENT]: Belt placed next to the miner")
                        else:  # ak ma belt v origine orientaciu kolidujucu s beltom v okoli
                            self.new_reward += -500
                            self.logger.info("[PLACEMENT]: Belt placed next to the miner but WRONG")
            # self.new_reward -= 50 * len(self.belts)
            # print("PLACEMENT: Belt placed at x:{} y:{}".format(self.player.x, self.player.y))
        elif action_idx < 20:  # INSERTER
            self.map[self.player.x][self.player.y][Channels.ENTITY] = Entities.INSERTER
            if action_idx == 16:
                self.map[self.player.x][self.player.y][Channels.UP] = 1
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.DOWN) != Entities.EMPTY:
                    self.new_reward += 100
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.UP) == Entities.CHEST:
                    self.new_reward += 1000
            elif action_idx == 17:
                self.map[self.player.x][self.player.y][Channels.DOWN] = 1
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.UP) != Entities.EMPTY:
                    self.new_reward += 100
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.DOWN) == Entities.CHEST:
                    self.new_reward += 1000
            elif action_idx == 18:
                self.map[self.player.x][self.player.y][Channels.LEFT] = 1
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.RIGHT) != Entities.EMPTY:
                    self.new_reward += 100
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.LEFT) == Entities.CHEST:
                    self.new_reward += 1000
            elif action_idx == 19:
                self.map[self.player.x][self.player.y][Channels.RIGHT] = 1
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.LEFT) != Entities.EMPTY:
                    self.new_reward += 100
                if self.get_map_oriented_increment(self.player.x, self.player.y, Channels.ENTITY,
                                                   Orientations.RIGHT) == Entities.CHEST:
                    self.new_reward += 1000
            self.inserters.append((self.player.x, self.player.y))
            # self.new_reward -= 50 * len(self.inserters)
            # print("PLACEMENT: Inserter placed at x:{} y:{}".format(self.player.x, self.player.y))

    def _place_invariant(self, action_idx):
        if action_idx == 20:  # CHEST
            self.map[self.player.x][self.player.y][Channels.ENTITY] = Entities.CHEST
            self.chests.append((self.player.x, self.player.y))
            # self.new_reward = -500 * len(self.chests)
        # print("PLACEMENT: Chest placed at x:{} y:{}".format(self.player.x, self.player.y))

    def __init__(self, level: LevelSelection, seed: int, frame_skip: int, human_control: bool,
                 custom_reward_threshold: Union[int, float], visualization_parameters: VisualizationParameters,
                 **kwargs):
        super().__init__(level, seed, frame_skip, human_control, custom_reward_threshold, visualization_parameters,
                         **kwargs)
        self.state_space = StateSpace(
            {"observation": PlanarMapsObservationSpace(np.array([self.map_height, self.map_width, len(Channels)]), low=0, high=20)}
        )
        self.init_env()
        self.action_space = DiscreteActionSpace(num_actions=21,
                                                descriptions={"0": "up", "1": "down", "2": "left", "3": "right",
                                                              "4": "up-left", "5": "up-right", "6": "down-left",
                                                              "7": "down-right", "8": "mine-up", "9": "mine-down",
                                                              "10": "mine-left", "11": "mine-right", "12": "belt-up",
                                                              "13": "belt-down", "14": "belt-left", "15": "belt-right",
                                                              "16": "inserter-up", "17": "inserter-down",
                                                              "18": "inserter-left", "19": "inserter-right",
                                                              "20": "chest"})

    def _take_action(self, action_idx: ActionType) -> None:
        self.new_reward = 0
        if action_idx < 8:
            self._move_player(action_idx)
        elif action_idx < 20:
            self._place_oriented(action_idx)
        else:
            self._place_invariant(action_idx)

    def _update_state(self) -> None:
        self.update_map()
        self.done = False
        self.reward = self.new_reward
        if self.done:
            pass
        elif self.total_reward_in_current_episode < self.minimum_reward:
            self.done = True
            self.logger.info("[TERMINATION]: Exceed minimum reward")
        elif len(self.chests) > 0:
            for chest in self.chests:
                if self.map[chest[0]][chest[1]][Channels.CONTENT] > 0:
                    self.done = True
                    self.reward = 3000
                    self.logger.info("[DONE]: CHEST AT x:{} y:{} CONTAINS RESULT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(chest[0], chest[1]))
                    break
        if self.current_episode_steps_counter > 100:
            self.done = True
            self.logger.info("[TERMINATION]: Current episode steps counter exceeded 5000")

        self.state = {"observation": self.map}

    def _restart_environment_episode(self, force_environment_reset=False) -> None:
        self.logger.info("[MAP UPDATE]: Miners count: {}".format(len(self.miners)))
        self.logger.info("[MAP UPDATE]: Belts count: {}".format(len(self.belts)))
        self.logger.info("[MAP UPDATE]: Inserters count: {}".format(len(self.inserters)))
        self.logger.info("[MAP UPDATE]: Chests count: {}".format(len(self.chests)))
        del self.map
        gc.collect()
        self.init_env()


class ControlSuiteEnvironmentParameters(EnvironmentParameters):
    def __init__(self):
        super().__init__()
        self.default_input_filter = NoInputFilter()
        self.default_output_filter = NoOutputFilter()

    @property
    def path(self):
        return 'rl_coach.environments.second_test:FactEnv'
