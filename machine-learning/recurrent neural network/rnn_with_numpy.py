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

    def cross_entropy(self, x, y):
        L = 0
        for i in xrange(len(y)):
            o, s = self.forward_propagation(x[i])

            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def train(self, x, y):
        N = np.sum((len(sentence) for sentence in y))
        return self.cross_entropy(x, y)/N

    def softmax(self, v, t=1.0):
        e = np.exp(v / t)
        return e / np.sum(e)

    def predict(self, x):
        o, x = self.forward_propagation(x)
        return np.argmax(o, axis=1)
