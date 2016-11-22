from glob import glob

import numpy as np
import os
import random
import logging

import shutil
import tensorflow as tf
import tflearn as tl
import re


class Agent(object):
    def __init__(self, env, replay, episode_n=50000000, gamma=0.95, gpu_memory_fraction=0.4):
        self.env = env
        self.replay = replay
        self._build_dq_network()
        self._build_optimizer_network()
        self._init_tensorflow(gpu_memory_fraction)

        # Train Configuration
        self.episode_n = episode_n
        self.pre_train_n = 1000
        self.train_frequency = self.replay.action_repeat
        self.target_update_step = 100  # 10000

        # Initialization
        self.step = 0
        self.loss = 0
        self.losses = []

        # Logger
        FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
        logging.basicConfig(format=FORMAT)
        # console_handler = logging.StreamHandler()
        self.logger = logging.getLogger('NeoDL')
        self.logger.setLevel(logging.DEBUG)
        # self.logger.addHandler(console_handler)

        # Epsilon
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
        input = tf.placeholder('float32', [None, self.replay.action_repeat, self.env.height, self.env.width],
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
        action_q_values = tf.reduce_sum(tf.mul(self.dqn_output, one_hot), reduction_indices=1)
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

    def _build_summaries(self):
        tags = ['global_step', 'epsilon', 'net_score', 'loss']

        self.summary_placeholders = {}
        self.summary_ops = {}
        for tag in tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
            self.summary_ops[tag] = tf.scalar_summary(tag, self.summary_placeholders[tag])

        if os.path.exists('/tmp/anderson_qlearning.tensorboard'):
            shutil.rmtree('/tmp/anderson_qlearning.tensorboard')
        self.writer = tf.train.SummaryWriter('/tmp/anderson_qlearning.tensorboard', self.sess.graph)

    def summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
            })
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, step)

    def _init_tensorflow(self, gpu_memory_faction=0.4):
        init = tf.initialize_all_variables()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_faction, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(init)

    def train(self):
        self._build_summaries()
        self.step = 0
        for global_step in xrange(self.episode_n):
            self.env.get_initial_states()
            self.losses = list()
            net_score = 0

            while True:
                self.step += 1
                # self.env.render()

                # Predict
                if self.replay.available:
                    screens = self.env.recent_screens()
                    action = self.predict([screens])
                else:
                    action = self.env.random_action()

                # Action!
                screen, reward, done, info = self.env.step(action)

                # Store the memory
                self.env.add_screeen(screen)
                self.replay.add(screen, action, reward, done)

                # Observe
                self.observe(screen, reward, action, done)

                # Store networks
                self.persist(global_step)

                # Logging
                net_score += reward

                if done:
                    break

            self.summary({'global_step': global_step,
                          'epsilon': self.epsilon,
                          'net_score': net_score,
                          'loss': np.mean(self.losses)}, self.step)

        self.env.close()

    def evaluate(self):
        self.restore()

        for _ in range(100):
            self.env.get_initial_states()

            net_reward = 0
            while True:
                self.env.render()
                screens = self.env.recent_screens()
                action = self.predict([screens], epsilon=0)
                screen, reward, done, info = self.env.step(action)

                net_reward += reward
                if done:
                    break

                # Store the memory
                self.env.add_screeen(screen)

            self.logger.info('Net Reward: %d' % net_reward)
            self.env.close()

    def persist(self, global_step):
        if self.step % self.save_step == 0:
            if not os.path.exists('_network'):
                os.mkdir('_network')

            [os.remove(os.path.join('_network', f)) for f in os.listdir('_network')]

            self.saver.save(self.sess, "_network/neo-dl.ckpt", global_step=global_step)
            self.logger.info('_network/neo-dl.ckpt has been persisted')

    def restore(self):

        filelist = glob('_network/neo-dl.ckpt*')
        saved_files = list()
        for f in filelist:
            match = re.search(r'neo-dl\.ckpt-(?P<number>\d+)$', f)
            if match:
                saved_files.append((int(match.group('number')), f))

        saved_files = sorted(saved_files, key=lambda x: -int(x[0]))
        if len(saved_files) >= 1:
            self.saver.restore(self.sess, saved_files[0][1])
            self.logger.info('%s has been restored', filelist[0])

    def predict(self, screens, epsilon=None):
        epsilon = epsilon if epsilon is not None else self.epsilon

        if random.random() < epsilon:
            action = self.env.random_action()
        else:
            action = np.argmax(self.sess.run(self.dqn_output, feed_dict={self.dqn_input: screens}), axis=1)

        return action

    def observe(self, state, reward, action, done):
        if self.step > self.pre_train_n:
            if self.step % self.train_frequency == 0:
                self.minibatch()

            if self.step % self.target_update_step == 0:
                self.update_target_network()

    def update_target_network(self):
        self.sess.run(self._update_target_network)
        self.logger.info('target network has been updated!')

    def minibatch(self):
        if not self.replay.available:
            return

        prestates, actions, rewards, poststates, terminals = self.replay.sample()
        clipped_rewards = np.clip(rewards, -1, 1)

        # Calculate Target Network
        target_actions = self.sess.run(self.target_output, feed_dict={self.target_input: poststates})
        target_output = (1 - terminals) * clipped_rewards + self.gamma * np.max(target_actions, axis=1)

        # Calculate Deep Q Network
        predicted_actions = np.argmax(self.sess.run(self.dqn_output, feed_dict={self.dqn_input: prestates}), axis=1)

        cost = self.optimizer['cost']
        grad_update = self.optimizer['grad_update']
        loss, _ = self.sess.run([cost, grad_update], feed_dict={self.dqn_input: prestates,
                                                                self.optimizer['a']: predicted_actions,
                                                                self.optimizer['y']: target_output})
        self.losses.append(loss)
        # self.logger.info('Optimizer ran stochastic gradient descent')

    @property
    def epsilon(self):
        s, e = self._epsilon_range
        return max(0., (s - e) * (self._epsilon_end - max(0., self.step - self.pre_train_n)) / self._epsilon_end)

    def close(self):
        self.sess.close()
        self.game.close()
