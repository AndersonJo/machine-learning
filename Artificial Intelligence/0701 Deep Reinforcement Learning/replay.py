"""
Modification of
https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/experience.py
"""
import numpy as np
import random


class ExperienceReplay(object):
    def __init__(self, env, action_repeat=4, batch_size=32, memory_size=500000):
        dims = list(env.dims)

        self.action_repeat = action_repeat
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.screens = np.empty([self.memory_size] + dims, dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty([self.batch_size, self.action_repeat] + dims, dtype=np.float16)
        self.poststates = np.empty([self.batch_size, self.action_repeat] + dims, dtype=np.float16)

        self.count = 0
        self.current = 0

    def add(self, screen, reward, action, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.action_repeat - 1:
            # use faster slicing
            return self.screens[(index - (self.action_repeat - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.action_repeat))]
            return self.screens[indexes, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.action_repeat
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.action_repeat, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.action_repeat < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.action_repeat):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals

    def retrieve(self, index=None):
        """
        Retrieve 4 screens (4 is action_repeat)
        """
        if index is None:
            index = min(self.count, self.memory_size)

        index = index % self.count
        if index >= self.action_repeat - 1:
            return self.screens[(index - (self.action_repeat - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.action_repeat))]
            return self.screens[indexes, ...]

    @property
    def available(self):
        return self.count >= self.action_repeat

    @property
    def size(self):
        return self.count


if __name__ == '__main__':
    ExperienceReplay()
