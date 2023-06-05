import copy

class GridWorld:
    def __init__(self):
        self.filed_type = {
            'N': 0, # normal
            'G': 1, # goal
            'W': 2, # wall
            'T': 3, # trap
        }

        self.actions = {
            'UP': 0,
            'DOWN': 1,
            'LEFT': 2,
            'RIGHT': 3,
        }

        self.map = [
            [3, 2, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 2, 0],
            [2, 0, 2, 0],
            [0, 0, 0, 0],
        ]

        self.start_pos = (0, 4) # start positon
        self.agent_pos = copy.deepcopy(self.start_pos) # current positon

    def step(self, action):
        to_x, to_y = copy.deepcopy(self.agent_pos)

        if not self._is_possible_action(to_x, to_y, action):
            return self.agent_pos, -1, False

        if action == self.actions['UP']:
            to_y += -1
        elif action == self.actions['DOWN']:
            to_y += 1
        elif action == self.actions['LEFT']:
            to_x += -1
        elif action == self.actions['RIGHT']:
            to_x += 1

        is_goal = self._is_end_episode(to_x, to_y)
        reward = self._compute_reward(to_x, to_y)
        self.agent_pos = (to_x, to_y)
        return self.agent_pos, reward, is_goal

    def _is_end_episode(self, x, y):
        if self.map[y][x] == self.filed_type['G']:
            return True
        elif self.map[y][x] == self.filed_type['T']:
            return True
        else:
            return False

    def _is_wall(self, x, y):
        if self.map[y][x] == self.filed_type['W']:
            return True
        else:
            return False

    def _is_possible_action(self, x, y, action):
        to_x = x
        to_y = y

        if action == self.actions['UP']:
            to_y += -1
        elif action == self.actions['DOWN']:
            to_y += 1
        elif action == self.actions['LEFT']:
            to_x += -1
        elif action == self.actions['RIGHT']:
            to_x += 1

        if len(self.map) <= to_y or 0 > to_y:
            return False
        elif len(self.map[0]) <= to_x or 0 > to_x:
            return False
        elif self._is_wall(to_x, to_y):
            return False

        return True

    def _compute_reward(self, x, y):
        if self.map[y][x] == self.filed_type['N']:
            return 0
        elif self.map[y][x] == self.filed_type['G']:
            return 100
        elif self.map[y][x] == self.filed_type['T']:
            return -100

    def reset(self):
        self.agent_pos = self.start_pos
        return self.start_pos
