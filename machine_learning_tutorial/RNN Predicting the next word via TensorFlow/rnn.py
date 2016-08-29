# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np


class AndersonRNN(object):
    def __init__(self, N, hidden=100):
        # self.tf_session = tf_session
        self.N = N
        self.hidden = hidden

        self.U = tf.Variable(self._random_init_weights([hidden, N]))
        self.V = tf.Variable(self._random_init_weights([N, hidden]))
        self.W = tf.Variable(self._random_init_weights([hidden, hidden]))

        self.S = tf.Variable()

        self.i = tf.Variable(0)
        self.i_n = tf.Variable(0)

    def _random_init_weights(self, dimension):
        """
        Initialize U, V, and W
        """
        return tf.random_uniform(dimension, -1 / np.sqrt(self.N), 1. / np.sqrt(self.N))

    def forward_propagation(self, x):
        """
        :param x: an array of input data
        """
        T = len(x)
        tf.assign(self.i, 0)
        tf.assign(self.i_n, tf.size(x))

        for t in xrange(T):
            tf.tanh(self.U[:, x[t]])

            print tf.size(x)
            print


def run():
    rnn = AndersonRNN(10000)

    x = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        rnn.forward_propagation([0, 122, 10, 30, 20])

        print np.array(sess.run([x])[0])
        print
        print np.array(sess.run([tf.gather_nd(x, [[0, 2], [1, 1], [2, 0]])])[0])


if __name__ == '__main__':
    run()
