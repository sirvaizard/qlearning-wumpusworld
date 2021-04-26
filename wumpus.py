import random
from typing import Tuple
from itertools import combinations


class WumpusWorld:

    def __init__(self):
        self.board = [
            ['.', 'P', '.', 'G'],
            ['.', '.', '.', '.'],
            ['.', 'W', '.', 'P'],
            ['.', '.', 'P', '.'],
        ]

        self.actions = ['turn left', 'turn right', 'move foward',
                        'grab', 'shoot arrow', 'climb']

        # Environment
        self.observation_space = 0
        self.action_space = 6
        self.width = 4
        self.height = 4

        # State
        # Player
        self.player_position = [self.height-1, 0]
        self.facing_north = False
        self.facing_south = False
        self.facing_west = True
        self.facing_east = False

        # Wumpus
        self.wumpus_position = (2, 1)

        # Perceptions
        self.wumpus_alive = True
        self.has_arrow = True
        self.has_gold = False
        self.done = False
        self.stench = False
        self.breeze = False
        self.glitter = False

        # Rewards
        self.death_reward = -1000
        self.step_reward = -1
        self.win_reward = 1000
        self.shoot_reward = -10
        self.kill_wumpus = 10

        self.possible_perceptions = ('stench', 'breeze', 'has_gold',
                                     'has_arrow', 'wumpus_alive', 'glitter',
                                     'facing_north', 'facing_south',
                                     'facing_west', 'facing_east')
        self.set_observation_space()

    @property
    def state(self) -> int:
        """ Return a int corresponding to the game current state"""
        state = self.player_position[0] * self.width + self.player_position[1]

        offset = 0

        def check_perceptions(perceptions: Tuple) -> bool:
            for percept in perceptions:
                if not self.__dict__[percept]:
                    return False
            return True

        for i, percept in enumerate(self.possible_perceptions):
            if check_perceptions(percept):
                offset = i+1
                break

        return state + (offset * self.width * self.height)

    def set_observation_space(self):
        """Set all possible state's variables combinations"""
        all_combinations = []
        for i in range(1, len(self.possible_perceptions)+1):
            for x in combinations(self.possible_perceptions, i):
                all_combinations.append(x)
        self.possible_perceptions = tuple(reversed(all_combinations))
        self.observation_space = len(
            all_combinations) * self.height * self.width

    def reset(self) -> int:
        """Reset the game state and return the initial state"""
        self.player_position = [self.height-1, 0]
        self.facing_east = False
        self.facing_north = False
        self.facing_south = False
        self.facing_west = True

        self.wumpus_alive = True
        self.done = False
        self.has_arrow = True
        self.has_gold = False
        self.reset_perceptions()

        return 0

    def reset_perceptions(self):
        self.breeze = False
        self.stench = False
        self.glitter = False

    def check_death(self) -> bool:
        """Return a True if the player's current position has a 
        wumpus or a pit and False otherwise.
        """
        x, y = self.player_position
        if ((self.wumpus_alive and self.wumpus_position == (x, y))
                or (self.board[x][y] == 'P')):
            self.done = True
            return True
        return False

    def step(self, action: int) -> Tuple[int, int, bool]:
        """Take and perform an action, and return a tuple
        with the new state, the reward received by that action
        and if the episode ir over or not.
        """
        if action > self.action_space:
            raise ValueError('Invalid action.')

        reward = self.step_reward

        if action == 0:  # turn left
            if self.facing_east:
                self.facing_east = False
                self.facing_south = True

            elif self.facing_north:
                self.facing_north = False
                self.facing_east = True

            elif self.facing_south:
                self.facing_south = False
                self.facing_west = True

            elif self.facing_west:
                self.facing_west = False
                self.facing_north = True

        elif action == 1:  # turn right
            if self.facing_east:
                self.facing_east = False
                self.facing_north = True

            elif self.facing_north:
                self.facing_north = False
                self.facing_west = True

            elif self.facing_south:
                self.facing_south = False
                self.facing_east = True

            elif self.facing_west:
                self.facing_west = False
                self.facing_south = True

        elif action == 2:  # move foward
            direction = [0, 0]
            if self.facing_south:
                direction[0] = 1
            elif self.facing_north:
                direction[0] = -1
            elif self.facing_east:
                direction[1] = -1
            else:
                direction[1] = 1

            x, y = [sum(x) for x in zip(direction, self.player_position)]

            if x >= 0 and x < self.height and y >= 0 and y < self.width:
                self.player_position = [x, y]
                if self.check_death():
                    reward = self.death_reward

        elif action == 3:  # grab
            x, y = self.player_position
            if not self.has_gold and self.glitter:
                self.has_gold = True

        elif action == 4:  # shoot arrow
            # TODO: if killed, wumpus should scream
            if self.has_arrow:
                if self.facing_south:
                    if (self.wumpus_position[1] == self.player_position[1]
                            and self.wumpus_position[0] > self.player_position[0]):
                        self.wumpus_alive = False

                elif self.facing_north:
                    if (self.wumpus_position[1] == self.player_position[1]
                            and self.wumpus_position[0] < self.player_position[0]):
                        self.wumpus_alive = False

                elif self.facing_east:
                    if (self.wumpus_position[0] == self.player_position[0]
                            and self.wumpus_position[1] < self.player_position[1]):
                        self.wumpus_alive = False
                else:
                    if (self.wumpus_position[0] == self.player_position[0]
                            and self.wumpus_position[1] > self.player_position[1]):
                        self.wumpus_alive = False
                self.has_arrow = False
                if not self.wumpus_alive:
                    # by giving a reward for killing the wumpus, we "guide" the
                    # agent to aways try to kill the wumpus, otherwise it'd avoid it.
                    reward = self.kill_wumpus
                else:
                    reward = self.shoot_reward

        elif action == 5:  # climb
            if self.player_position == [self.height-1, 0]:
                if self.has_gold:
                    reward = self.win_reward
                else:
                    reward = -100
                self.done = True

        # Set perceptions
        x, y = self.player_position

        self.reset_perceptions()

        if self.board[x][y] == 'P':
            self.breeze = True

        if self.board[x][y] == 'W':
            self.stench = True

        if self.board[x][y] == 'G':
            self.glitter = True

        return self.state, reward, self.done

    def render(self):
        x, y = self.player_position
        wx, wy = self.wumpus_position
        board = [list(line) for line in self.board]
        if not self.wumpus_alive:
            board[wx][wy] = '-'
        if self.has_gold:
            board[0][3] = '.'
        board[x][y] = '@'
        for line in board:
            print(line)
        print()
