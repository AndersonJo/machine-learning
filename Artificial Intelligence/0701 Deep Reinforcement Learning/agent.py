from glob import glob

import numpy as np
import os
import random
import logging

import shutil
from time import sleep

import tensorflow as tf
import tflearn as tl
import re


class Agent(object):
    def __init__(self, env, replay, episode_n=50000000, gamma=0.95, gpu_memory_fraction=0.5):
        self.env = env
        self.replay = replay
        self._build_dq_network()
        self._build_optimizer_network()
        self._init_tensorflow(gpu_memory_fraction)

        # Train Configuration
        self.episode_n = episode_n
        self.pre_train_n = 5000
        self.train_frequency = self.replay.action_repeat
        self.target_update_step = 10000

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

        # Persist
        self.persist_step = 50  # Global Step
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
        input = tf.placeholder('float', [None, self.replay.action_repeat, self.env.height, self.env.width],
                               name=name + '_input')

        transposed = tf.transpose(input, [0, 2, 3, 1])
        net1 = tl.conv_2d(transposed, 32, 8, strides=4, activation='relu', name=name + '_cnn1')
        net2 = tl.conv_2d(net1, 64, 4, strides=2, activation='relu', name=name + '_cnn2')
        # net3 = tl.conv_2d(net2, 64, 3, strides=1, activation='relu', name=name + '_cnn3')

        net3 = tl.fully_connected(net2, 256, activation='relu', name=name + '_net1')
        output = tl.fully_connected(net3, self.env.action_size, name=name + '_output')
        return {
            'input': input,
            'transposed': transposed,
            'net1': net1,
            'net2': net2,
            'net3': net3,
            # 'net4': net4,
            'output': output
        }

    def _build_optimizer_network(self):
        # Define cost and gradient update op
        y = tf.placeholder("float", [None], name='optimizer_y')
        a = tf.placeholder("float", [None, self.env.action_size], name='optimizer_a')

        dqn_mt = tf.mul(self.dqn_output, a)
        action_q_values = tf.reduce_sum(dqn_mt, reduction_indices=1)
        # delta = y - action_q_values
        cost = tl.mean_square(action_q_values, y)
        # clipped_delta = tf.clip_by_value(delta, -1, 1)
        # cost = tf.reduce_mean(tf.square(clipped_delta))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1, momentum=0.95, epsilon=0.01)
        grad_update = optimizer.minimize(cost, var_list=self.network_variables)

        # regression = regression(net, optimizer=rmsprop)

        self.optimizer = {
            'a': a,
            'y': y,
            'action_q_values': action_q_values,
            # 'delta': delta,
            # 'clipped_delta': clipped_delta,
            'cost': cost,
            'optimizer': optimizer,
            'grad_update': grad_update
        }

    def _build_summaries(self):
        tags = ['global_step', 'epsilon', 'net_score', 'loss', 'replay_memory_size',
                'dqn_net2', 'dqn_net3', 'dqn_net4',
                'target_net2', 'target_net3', 'target_net4']

        self.summary_placeholders = {}
        self.summary_ops = {}
        for tag in tags:
            self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
            self.summary_ops[tag] = tf.scalar_summary(tag, self.summary_placeholders[tag])

        if os.path.exists('/tmp/anderson_qlearning.tensorboard'):
            shutil.rmtree('/tmp/anderson_qlearning.tensorboard')
        self.writer = tf.train.SummaryWriter('/tmp/anderson_qlearning.tensorboard', self.sess.graph)

    def _build_image_summaries(self):
        dqn = tf.image_summary('dqn network', self.dqn['transposed'], max_images=4)
        target = tf.image_summary('target network', self.target['transposed'], max_images=4)

        self.image_summaries = {
            'dqn': dqn,
            'target': target
        }

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
        self._build_image_summaries()

        self.step = 0
        for global_step in xrange(self.episode_n):
            self.env.get_initial_states()
            self.losses = list()
            net_score = 0

            screens = self.env.recent_screens()

            while True:
                self.step += 1
                # self.env.render()

                # Predict
                screens = self.env.recent_screens()
                actions = self.predict([screens])
                action = np.argmax(actions)

                # Action!
                screen, reward, done, info = self.env.step(action)

                # Store the memory
                self.replay.add(screen, reward, action, done)

                # Observe
                self.observe(screen, reward, action, done)

                # Logging
                net_score += reward

                if done:
                    break

            dqn_net2 = np.mean(self.sess.run(self.dqn['net2'].W))
            # dqn_net3 = np.sum(self.sess.run(self.dqn['net3'].W))

            target_net2 = np.mean(self.sess.run(self.target['net2'].W))
            # target_net3 = np.mean(self.sess.run(self.target['net3'].W))

            self.summary({'global_step': global_step,
                          'epsilon': self.epsilon,
                          'net_score': net_score,
                          'loss': np.mean(self.losses, dtype='float'),
                          'replay_memory_size': self.replay.size,
                          'dqn_net2': dqn_net2,
                          # 'dqn_net3': dqn_net3,
                          # 'dqn_net4': dqn_net4,
                          'target_net2': target_net2,
                          # 'target_net3': target_net3,
                          # 'target_net4': target_net4
                          }, self.step)

            if global_step % self.persist_step == 0 and global_step > 10:
                self.persist(global_step)

            self.env.close()

    def evaluate(self):
        for _ in range(100):
            self.env.get_initial_states()

            net_reward = 0
            while True:
                self.env.render()
                screens = self.env.recent_screens()
                actions = self.predict([screens], epsilon=0.1)

                action = np.argmax(actions)
                screen, reward, done, info = self.env.step(action)

                net_reward += reward
                if done:
                    break

                # Store the memory
                self.env.add_screeen(screen)

            self.logger.info('Net Reward: %d' % net_reward)
            self.env.close()

    def persist(self, global_step):
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
            self.logger.info('%s has been restored', saved_files[0][1])

    def predict(self, screens, epsilon=None):
        epsilon = epsilon if epsilon is not None else self.epsilon

        actions = np.zeros(self.env.action_size)
        if random.random() < epsilon:
            action_index = self.env.random_action()
        else:
            action_index = np.argmax(self.sess.run(self.dqn_output, feed_dict={self.dqn_input: screens}), axis=1)

        actions[action_index] = 1
        return actions

    def observe(self, state, reward, action, done):
        if self.step > self.pre_train_n:
            self.minibatch()

            if self.step % self.target_update_step == 0:
                self.update_target_network()

    def update_target_network(self):
        self.sess.run(self._update_target_network)
        self.logger.info('target network has been updated!')

    def minibatch(self):
        if not self.replay.available:
            return

        prestates, stored_actions, rewards, poststates, terminals = self.replay.sample()

        # Calculate Target Network
        img_target = self.image_summaries['target']
        target_actions, target_summary = self.sess.run([self.target_output, img_target],
                                                       feed_dict={self.target_input: poststates})

        clipped_rewards = np.clip(rewards, -1., 1.)
        target_output = clipped_rewards + self.gamma * np.max(target_actions, axis=1)

        # Calculate Deep Q Network (Predict)
        # predicted_actions = self.sess.run(self.dqn_output, feed_dict={self.dqn_input: prestates})
        # action_indices = np.argmax(predicted_actions, axis=1)
        # actions = np.zeros(predicted_actions.shape)
        # actions[range(len(actions)), action_indices] = 1

        actions = np.zeros((len(stored_actions), self.env.action_size))
        for i, action in enumerate(stored_actions):
            actions[i, action] = 1


        # Optimize
        # 'dqn_mt': dqn_mt,
        # 'action_q_values': action_q_values,
        # 'delta': delta,
        # 'clipped_delta': clipped_delta,
        # 'cost': cost,
        cost = self.optimizer['cost']
        grad_update = self.optimizer['grad_update']

        img_dqn = self.image_summaries['dqn']

        dqn_summary, loss, _ = self.sess.run([img_dqn, cost, grad_update],
                                             feed_dict={self.dqn_input: prestates,
                                                        self.optimizer['a']: actions,
                                                        self.optimizer['y']: target_output})


        self.writer.add_summary(dqn_summary)
        self.writer.add_summary(target_summary)
        self.losses.append(loss)

    @property
    def epsilon(self):
        s, e = self._epsilon_range
        return max(0., (s - e) * (self._epsilon_end - max(0., self.step - self.pre_train_n)) / self._epsilon_end)

    def close(self):
        self.sess.close()
        self.game.close()
