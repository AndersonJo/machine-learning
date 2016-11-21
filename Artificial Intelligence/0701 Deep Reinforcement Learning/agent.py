import random

import tensorflow as tf
import tflearn as tl


class Agent(object):
    def __init__(self, env, replay, replay_size=4, gpu_memory_fraction=0.4):
        self.env = env
        self.replay = replay
        self._build_training_network()
        self._build_target_network()
        self._init_tensorflow(gpu_memory_fraction)

        # ETC Configurations
        self.pre_train_n = 100
        self.train_frequency = self.replay.replay_size

        # Epsilon
        self.step = 0
        self._epsilon_range = (1., 0.1)
        self._epsilon_end = 5000000

    def _build_training_network(self):
        #######################
        # Training Network
        #######################
        self.dq_states = tf.placeholder('float32',
                                        [None, self.replay.replay_size, self.env.height, self.env.width])

        # Convolutional Neural Network
        net = tl.conv_2d(self.dq_states, 32, 8, strides=4, activation='relu')
        net = tl.conv_2d(net, 64, 4, strides=2, activation='relu')
        net = tl.conv_2d(net, 64, 3, strides=1, activation='relu')

        # Deep Neural Network
        net = tl.fully_connected(net, 512, activation='relu', name='l4')
        q_values = tl.fully_connected(net, self.env.action_size, name='q')
        self.dq_states_action = tf.argmax(q_values, dimension=1)

    def _build_target_network(self):
        #######################
        # Target Network
        #######################
        self.target = tf.placeholder('float32',
                                     [None, self.replay.replay_size, self.env.height, self.env.width],
                                     name='target')

        target_net = tl.conv_2d(self.target, 32, 8, strides=4, activation='relu')
        target_net = tl.conv_2d(target_net, 64, 4, strides=2, activation='relu')
        target_net = tl.conv_2d(target_net, 64, 3, strides=1, activation='relu')

        target_net = tl.fully_connected(target_net, 512, activation='relu', name='target_l4')
        self.target_action = tl.fully_connected(target_net, self.env.action_size, name='target_q')

        self.target_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.target_nd = tf.gather_nd(self.target_action, self.target_idx)  # Matrix Indexing

    def _init_tensorflow(self, gpu_memory_faction=0.4):
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_faction, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init)

    def train(self, epoch=100):
        state = self.env.reset()
        buf_count = 0
        self.step = 0

        while True:
            self.step += 1
            self.env.render()

            # Predict
            action = self.predict(state)

            # Action!
            state, reward, done = self.env.step(action)

            # Store the memory
            self.replay.add(state, action, reward, done)

            # Observe
            self.observe(state, reward, action, done)

            break
        self.env.close()

    def observe(self, state, reward, action, done):
        if self.step > self.pre_train_n:
            if self.step % self.train_frequency == 0:
                self.minibatch()

    def predict(self, state, epsilon=None):
        epsilon = epsilon if epsilon else self.epsilon

        if random.random() < epsilon:
            action = self.env.random_action()
        else:
            action = self.sess.run(self.dq_states,
                                   feed_dict={self.dq_states: state})  # self.replay.sample(self.replay.sample())
        return action

    def minibatch(self):
        if not self.replay.available:
            self.

    @property
    def epsilon(self):
        s, e = self._epsilon_range
        return max(0., (s - e) * (self._epsilon_end - max(0., self.step - self.pre_train_n)) / self._epsilon_end)

    def close(self):
        self.sess.close()
        self.game.close()
