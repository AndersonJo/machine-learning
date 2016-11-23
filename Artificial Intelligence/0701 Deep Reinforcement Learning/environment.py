from collections import deque

import gym
import cv2
import numpy as np


class Environment(object):
    def __init__(self, game, action_repeat=4):
        self.game = gym.make(game)
        self.action_size = self.game.action_space.n
        self.action_repeat = action_repeat
        self.height, self.width = self.dims = (84, 84)

        self._buffer = deque(maxlen=self.action_repeat)

    def play_sample(self):
        while True:
            self.game.render()
            action = self.game.action_space.sample()
            observation, reward, done = self.step(action)
            if done:
                break
        self.game.close()

    def step(self, action, gray=True, resize=True):
        screen, reward, done, info = self.game.step(action)
        screen = self.preprocess(screen)

        self.add_screeen(screen)

        return screen, reward, done, info

    def preprocess(self, screen):
        preprocessed = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY) / 255., (self.height, self.width))
        return preprocessed

    def get_initial_states(self):
        screen = self.game.reset()
        screen = self.preprocess(screen)
        screens = np.stack([screen for _ in range(self.action_repeat)], axis=0)

        self._buffer = deque(maxlen=self.action_repeat)
        for _ in range(self.action_repeat):
            self._buffer.append(screen)
        return screens

    def recent_screens(self):
        return np.array(self._buffer)

    def add_screeen(self, screen):
        self._buffer.append(screen)

    def random_step(self, gray=True):
        action = self.game.action_space.sample()
        return self.step(action, gray=True)

    def random_action(self):
        return self.game.action_space.sample()

    def render(self):
        return self.game.render()

    def reset(self):
        return self.game.reset()

    def close(self):
        return self.game.close()
