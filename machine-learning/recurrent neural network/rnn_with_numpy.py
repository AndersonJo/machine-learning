import numpy as np


class RNNNumpy(object):
    def __init__(self, word_dim, hidden_dim=100):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.U = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (hidden_dim, word_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))

        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        for t in xrange(T):
            s[t] = self.U[:, x[t]]
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]

    def softmax(self, v, t=1.0):
        e = np.exp(v / t)
        return e/np.sum(e)

    def predict(self, x):
        o, x = self.forward_propagation(x)
        return np.argmax(o, axis=1)
