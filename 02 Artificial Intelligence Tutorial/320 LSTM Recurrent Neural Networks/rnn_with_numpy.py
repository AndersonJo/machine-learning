# -*- coding:utf-8 -*-
from datetime import datetime
from random import randint

import numpy as np


class RNNNumpy(object):
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        self.U = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        """
        :param x: 한문장이 되는.. 정수값을 갖는 vector
        :return:
        """
        # The total number of time steps
        T = len(x)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))

        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))

        for t in xrange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = self.softmax(self.V.dot(s[t]))

        return [o, s]

    def cross_entropy(self, x, y):
        """
        :param x: sentence
        """
        N = len(y)
        o, s = self.forward_propagation(x)
        predicted_output = o[np.arange(len(y)), y]
        L = np.sum(np.log(predicted_output))
        return -1 * L / N

    def bptt(self, x, y):
        """
        :param x: an array of a sentence
        """
        T = len(y)
        o, s = self.forward_propagation(x)
        dldU = np.zeros(self.U.shape)
        dldV = np.zeros(self.V.shape)
        dldW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(T), y] -= 1.

        for t in np.arange(T)[::-1]:
            dldV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dldW += np.outer(delta_t, s[bptt_step - 1])
                dldU[:, x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return dldU, dldV, dldW

    def calculate_gradients(self, x, y, learning_rate=0.005):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def train(self, x_train, y_train, learning_rate=0.005, npoch=100):
        N = len(y_train)
        loss_show = N / 10

        print
        'Start Training'
        print
        'Total Data: ', N

        for i in xrange(npoch):
            # One SGD step
            rand_idx = randint(0, N)
            self.calculate_gradients(x_train[rand_idx], y_train[rand_idx], learning_rate)
            if i % 100 == 0:
                self._print_error(self.cross_entropy(x_train[i], y_train[i]), i)

    @staticmethod
    def _print_error(cost, i):
        t = datetime.now()
        print
        '{time}: i={i}, cost={cost}'.format(time=t, i=i, cost=cost)

    def softmax(self, v, t=1.0):
        e = np.exp(v / t)
        return e / np.sum(e)

    def predict(self, x):
        o, x = self.forward_propagation(x)
        return np.argmax(o, axis=1)
