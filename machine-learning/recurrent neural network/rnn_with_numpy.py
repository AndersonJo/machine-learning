# -*- coding:utf-8 -*-
import numpy as np


class RNNNumpy(object):
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        
        self.U = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-1. / np.sqrt(word_dim), 1. / np.sqrt(word_dim), (hidden_dim, word_dim))
        print self.U.shape
        print self.V.shape
        print self.W.shape

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
        return self.cross_entropy(x, y) / N

    def bptt(self, x, y):
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
            delta_t = self.V.T.dot(delta_o[t] * (1 - (s[t] ** 2)))

            # Backpropagration through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dldW += np.outer(delta_t, s[bptt_step - 1])
                dldU[: x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return dldU, dldV, dldW

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                    np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error and gt and error_threshold:
                    print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gradient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed." % (pname)

    def softmax(self, v, t=1.0):
        e = np.exp(v / t)
        return e / np.sum(e)

    def predict(self, x):
        o, x = self.forward_propagation(x)
        return np.argmax(o, axis=1)
