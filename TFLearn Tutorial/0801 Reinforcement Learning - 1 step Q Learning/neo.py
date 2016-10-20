# -*- coding:utf-8 -*-
from collections import deque
import gym
import tflearn
import tensorflow as tf
#
#
# class AtariEnvironment(object):
#     def __init__(self, game, action_repeat=4, num_states=3):
#         """
#         :param game <str>: Game name
#         """
#         assert game is not None
#
#         self.env = gym.make(game)  # Create an GYM environment
#         self.num_actions = self.env.action_space.n  # actions의 갯수
#         self.states = deque(maxlen=num_states)
#         self.action_repeat = action_repeat
#
#
# class Agent(object):
#     def __init__(self, env):
#         self.env = env
#
#     def build_dqn(self):
#         action_repeat = self.env.action_repeat
#         num_actions = self.env.num_actions
#
#         # Input Shape: [batch, channel, height, width, colors]
#         inputs = tf.placeholder(tf.float32, [None, action_repeat, 84, 84, 3])
#         print tf.shape(inputs)
#         net = tflearn.conv_3d(inputs, 64, 4, strides=4, activation='relu')
#         net = tflearn.conv_2d(inputs, 64, 4, strides=4, activation='relu')
#         net = tflearn.fully_connected(net, 256, activation='relu')
#         q_values = tflearn.fully_connected(net, num_actions)
#         return inputs, q_values
#
#
# if __name__ == '__main__':
#     GAME = 'MsPacman-v0'
#     env = AtariEnvironment(GAME)
#     agent = Agent(env)
#     agent.build_dqn()
#
