"""
Modification of
https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/experience.py
"""
import numpy as np
import random


class ExperienceReplay(object):
    def __init__(self, env, history_size=4, batch_size=32, memory_size=1000000):
        dims = list(env.dims)
        print 'replay dims:', dims
        self.history_size = history_size
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.screens = np.empty([self.memory_size] + dims, dtype=np.uint8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty([self.batch_size, self.history_size] + dims, dtype=np.float16)
        self.poststates = np.empty([self.batch_size, self.history_size] + dims, dtype=np.float16)

        self.count = 0
        self.current = 0

    def add(self, screen, reward, action, terminal):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):
        assert self.count >= self.history_size, 'Add more data'

        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.history_size, self.count - 1)
                if index >= self.current and index - self.history_size < self.current:
                    continue
                if self.terminals[(index - self.history_size):index].any():
                    continue
                break
            print 'len(indexes)', self.prestates.shape, self.retreive(index).shape
            self.prestates[len(indexes), ...] = self.retreive(index - 1)
            self.poststates[len(indexes), ...] = self.retreive(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        return self.prestates, actions, rewards, self.poststates, terminals

    def retreive(self, index):
        """
        Retrieve 4 screens (4 is history_size)
        """

        index = index % self.count
        if index >= self.history_size - 1:
            return self.screens[(index - (self.history_size - 1)):(index + 1), ...]
        else:
            indexes = [(index - i) % self.count for i in reversed(range(self.history_size))]
            return self.screens[indexes, ...]


if __name__ == '__main__':
    ExperienceReplay()
