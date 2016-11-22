import numpy as np
import os
import random

import tensorflow as tf
import tflearn as tl


class Agent(object):
    def __init__(self, env, replay, episode_n=100, gamma=0.95, gpu_memory_fraction=0.4):
        self.env = env
        self.replay = replay
        self._build_dq_network()
        self._build_optimizer_network()
        self._init_tensorflow(gpu_memory_fraction)

        # Train Configuration
        self.episode_n = episode_n
        self.pre_train_n = 100
        self.train_frequency = self.replay.history_size
        self.target_update_step = 100  # 10000

        # Epsilon
        self.step = 0
        self._epsilon_range = (1., 0.1)
        self._epsilon_end = 5000000

        # Gamma
        self.gamma = gamma  # used for discounted value

        # Saver
        self.save_step = 2000
        self.saver = tf.train.Saver(max_to_keep=5)

    def _build_dq_network(self):
        #######################
        # Training Network
        #######################
        existing_variables_length = len(tf.trainable_variables())
        self.dqn = self._build_dqn(name='dqn')
        self.dqn_input, self.dqn_output = self.dqn['input'], self.dqn['output']
        self.network_variables = tf.trainable_variables()[existing_variables_length:]

        #######################
        # Target Network
        #######################
        self.target = self._build_dqn(name='target')
        self.target_input, self.target_output = self.target['input'], self.target['output']
        self.target_network_variables = tf.trainable_variables()[
                                        len(self.network_variables) + existing_variables_length:]

        # Target Update Tensor
        self._update_target_network = [self.target_network_variables[i].assign(self.network_variables[i])
                                       for i in range(len(self.network_variables))]

    def _build_dqn(self, name):
        input = tf.placeholder('float32', [None, self.replay.history_size, self.env.height, self.env.width],
                               name=name + '_input')

        net1 = tl.conv_2d(input, 32, 8, strides=4, activation='relu', name=name + '_cnn1')
        net2 = tl.conv_2d(net1, 64, 4, strides=2, activation='relu', name=name + '_cnn2')
        net3 = tl.conv_2d(net2, 64, 3, strides=1, activation='relu', name=name + '_cnn3')

        net4 = tl.fully_connected(net3, 512, activation='relu', name=name + '_net1')
        output = tl.fully_connected(net4, self.env.action_size, name=name + '_output')
        return {
            'input': input,
            'net1': net1,
            'net2': net2,
            'net3': net3,
            'net4': net4,
            'output': output
        }

    def _build_optimizer_network(self):

        # Define cost and gradient update op
        y = tf.placeholder("float", [None], name='optimizer_y')
        a = tf.placeholder("int64", [None], name='optimizer_a')  # self.dqn_input

        one_hot = tf.one_hot(a, self.env.action_size, 1.0, 0.0)
        action_q_values = tf.reduce_sum(self.dqn_output * one_hot, reduction_indices=1)
        cost = tl.mean_square(action_q_values, y)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.05, momentum=0.95, epsilon=0.01)
        grad_update = optimizer.minimize(cost, var_list=self.network_variables)

        self.optimizer = {
            'a': a,
            'y': y,
            'one_hot': one_hot,
            'action_q_values': action_q_values,
            'cost': cost,
            'optimizer': optimizer,
            'grad_update': grad_update
        }

        # self.optimizer_target_input = tf.placeholder('float32', [None],
        #                                              name='optimizer_target_input')  # <- self.target_output
        # self.optimizer_dqn_input = tf.placeholder('int64', [None], name='optimizer_dqn_input')  # <- self.dqn_actions
        #
        # one_hot = tf.one_hot(self.optimizer_dqn_input, self.env.action_size, 1.0, 0.0)
        # q_acted = tf.reduce_sum(self.dqn_output * one_hot, reduction_indices=1)
        #
        # self.delta = self.optimizer_target_input - q_acted
        # self.loss = tf.reduce_mean(tf.square(self.delta))
        #
        # self.optim = tf.train.RMSPropOptimizer(learning_rate=0.01, momentum=0.95, epsilon=0.01).minimize(self.loss)

        # rmsprop = tl.optimizers.RMSProp(learning_rate=0.1, decay=0.9, epsilon=0.05)
        # self.regression = tl.regression(self.loss, optimizer=rmsprop)

    def _build_summaries(self):
        epsilon = tf.Variable(0.)
        tf.scalar_summary('Epsilon', epsilon)

        summaries = [epsilon]
        placeholders = [tf.placeholder("float") for _ in range(len(summaries))]
        summary_op = tf.merge_all_summaries()

        return {
            'summaries': summaries,
            'placeholders': placeholders,
            'summary_op': summary_op
        }

    def _init_tensorflow(self, gpu_memory_faction=0.4):
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_faction, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init)

    def train(self):

        buf_count = 0
        self.step = 0

        for global_step in xrange(self.episode_n):
            screen = self.env.reset()

            while True:
                self.step += 1
                self.env.render()

                # Predict
                if self.replay.available:
                    screens = self.replay.retrieve()  # Recent 4 screens
                    action = self.predict([screens])
                else:
                    action = self.env.random_action()

                # Action!
                screen, reward, done, info = self.env.step(action)

                # Store the memory
                self.replay.add(screen, action, reward, done)

                # Observe
                self.observe(screen, reward, action, done)

                if self.step % self.save_step == 0:
                    self.saver.save(self.sess, "/tmp/qlearning.ckpt", global_step=global_step)

                if done:
                    break

        self.env.close()

    def restore(self):
        if os.path.exists('/tmp/qlearning.ckpt'):
            self.saver.restore(self.sess, '/tmp/qlearning.ckpt')

    def predict(self, screen, epsilon=None):
        epsilon = epsilon if epsilon else self.epsilon

        if random.random() < epsilon:
            action = self.env.random_action()
        else:
            action = np.argmax(self.sess.run(self.dqn_output, feed_dict={self.dqn_input: screen}), axis=1)
        return action

    def observe(self, state, reward, action, done):
        if self.step > self.pre_train_n:
            if self.step % self.train_frequency == 0:
                self.minibatch()

            if self.step % self.target_update_step == 0:
                self.update_target_network()

    def update_target_network(self):

        self.sess.run(self._update_target_network)

    def minibatch(self):
        if not self.replay.available:
            return

        prestates, actions, rewards, poststates, terminals = self.replay.sample()

        # Calculate Target Network
        target_predicted_actions = self.sess.run(self.target_output, feed_dict={self.target_input: poststates})
        target_output = (1 - terminals) * rewards + self.gamma * np.max(target_predicted_actions, axis=1)

        # Calculate Deep Q Network
        predicted_actions = np.argmax(self.sess.run(self.dqn_output, feed_dict={self.dqn_input: prestates}), axis=1)

        # Optimization
        # loss, dqn, opt_v = self.sess.run([self.loss, self.dqn_output, self.optim],
        #                                  feed_dict={self.optimizer_target_input: target_output,
        #                                             self.optimizer_dqn_input: predicted_actions,
        #                                             self.dqn_input: prestates})
        # print loss, dqn

        cost = self.optimizer['cost']
        grad_upgrade = self.optimizer['grad_update']
        loss_value, _ = self.sess.run([cost, grad_upgrade], feed_dict={self.dqn_input: prestates,
                                                                       self.optimizer['a']: predicted_actions,
                                                                       self.optimizer['y']: target_output})

    @property
    def epsilon(self):
        s, e = self._epsilon_range
        return max(0., (s - e) * (self._epsilon_end - max(0., self.step - self.pre_train_n)) / self._epsilon_end)

    def close(self):
        self.sess.close()
        self.game.close()
