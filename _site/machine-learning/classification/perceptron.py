import numpy as np


class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=10):
        """
        :param eta: Learning Rate (between 0.0 and 1.0)
        :param n_iter: Training Dataset.
        """
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit Training Data
        """

        self._w = np.zeros(1 + X.shape[1])
        self._errors = []

        for i in range(self.n_iter):
            errors = 0
            for x, target in zip(X, y):
                update = self.eta * (target - self.predict(x))
                self._w[1:] += update * x
                self._w[0] += update
                errors += int(update != 0.0)

            self._errors.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self._w[1:]) + self._w[0]

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)
