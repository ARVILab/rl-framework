import numpy as np


class Goal(object):

    def __init__(self, low_goal=0.4, high_goal=0.5, step=0.01):

        """
        :param low_goal: float, lower diapason value of goal
        :param high_goal: float, higher diapason diapason value of goal
        :param step: size step between goals

        So MountCar observations presented as array size 2, where first value- position, second- velocity.
        We map all possible points of this environment with some precision (self._decimals), and map it as one-hot
        vector to pass it on NN. This precision we also need, in reason working with floating numbers, and mapping them
        to hand-crafted env will cause errors

        self._goals - set of goals than NN' ll try to reach
        self._env_map - hand_crafted env, that mapped all points with some precision
        self._map_step - for mapping position of agent to one-hot vector

        """

        self._size_step = 0.1 / step
        self._low_goal = low_goal
        self._high_goal = high_goal
        self._step = step
        self._decimals = len(str(int(self._size_step)))

        self._goals = np.arange(low_goal, high_goal, step)
        self._env_map = np.arange(-1.2, 0.6 + step, step)
        self._env_map = np.round(self._env_map, decimals=self._decimals)

        self._map_step = {}

        # unefficient
        self._init_map_step()

    def _init_map_step(self):

        slice_step = 0

        for i in self._env_map:

            self._map_step[i] = slice_step
            slice_step += 1

    ######################## Getters ########################

    def get_goals(self):

        return self._goals

    def get_size(self):

        return self._goals.size

    def get_env_map(self):

        return self._env_map

    def get_env_map_size(self):

        return self._env_map.size

    def get_random_goal_idx(self):

        return (np.random.choice(np.arange(self._goals.size)))

    def get_goal_by_idx(self, idx):

        assert idx < self._goals.size

        return (self._goals[idx])

    def get_idx_by_goal(self, goal):

        assert goal in self._map_step

        return (self._map_step[goal])

    def get_decimals(self):

        return self._decimals

    ######################## Vector encoding ########################

    def one_hot_goal(self, idx):

        assert idx >= 0
        assert idx < len(self._map_step)

        mask = [0] * self._env_map.size

        low_slice, high_slice = self._map_step[self._low_goal], self._map_step[self._high_goal]

        if idx < self._goals.size:
            masked_idx = [0 if i != idx else 1 for i in range(self._goals.size)]
            mask[low_slice:high_slice] = masked_idx
        else:
            mask[idx] = 1
        mask = np.expand_dims(np.array(mask), axis=0)

        return mask

    def one_hot_current_state(self, current_state):

        current_state_pos = round(current_state[0], self._decimals)

        mask = [0] * (self._env_map.size + 1)

        idx = np.argwhere(self._env_map == current_state_pos)[0, 0]
        mask[idx] = 1

        mask[-1] = current_state[1] #our velocity
        mask = np.expand_dims(np.array(mask), axis=0)

        return mask
