"""
Modification of
https://github.com/carpedm20/deep-rl-tensorflow/blob/master/agents/experience.py
"""
import numpy as np
import random


class ExperienceReplay(object):
    def __init__(self, env, action_repeat=4, batch_size=32, memory_size=1000000):
        dims = list(env.dims)

        self.action_repeat = action_repeat
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.int8)
        self.screens = np.empty([self.memory_size] + dims, dtype=np.uint8)
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

    def sample(self):
        assert self.count >= self.action_repeat, 'Add more data'

        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(self.action_repeat, self.count - 1)
                if index >= self.current and index - self.action_repeat < self.current:
                    continue
                if self.terminals[(index - self.action_repeat):index].any():
                    continue
                break

            self.prestates[len(indexes), ...] = self.retrieve(index - 1)
            self.poststates[len(indexes), ...] = self.retrieve(index)
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


if __name__ == '__main__':
    ExperienceReplay()
