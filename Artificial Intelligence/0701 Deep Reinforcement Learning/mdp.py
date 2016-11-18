# -*- coding:utf-8 -*-
import numpy as np


class ValueIteration(object):
    """
    Value Iteration Algorithm of Markov Decision Process with Python
    """

    def __init__(self, env, start=(0, 0)):
        """

        :param env: 2 dimensional arrays of Numpy
        :param start: 시작지점. 예를 들어 (0, 0)

        """
        self.env = env
        self.start = start
        self.actions = ()

    def transition(self, state, next_state, action):
        pass

    def init_state(self):
        pass

    def reward(self, state):
        pass

    def discounted_future_reward(self, state):
        pass

    def value_iteration(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
