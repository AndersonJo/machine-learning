from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import MultiRNNCell

# Load Data
data = pd.read_csv('../../data/time-series/international-airline-passengers.csv',
                   names=['passenger'],
                   skiprows=1, usecols=[1])

# Normalize Data using Min-Max Scaler
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
data = data.astype('float32')

# Seperate Train and Test Data
_size = int(len(data) * 0.7)  # 144
train, test = data[:_size], data[_size:]  # 100, 44


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i])
        dataY.append(dataset[i + 1])
    return np.array(dataX), np.array(dataY)


def reshape(dataset):
    return np.reshape(dataset, (dataset.shape[0], 1, dataset.shape[1]))


look_back = 1
train_x, train_y = create_dataset(train, look_back)
test_x, test_y = create_dataset(test, look_back)

train_x = reshape(train_x)
train_y = reshape(train_y)
test_x = reshape(test_x)
test_y = reshape(test_y)


# Model
# hidden_size = 1

# scope_name = 'test' + str(np.random.randint(0, 100000))
# with tf.variable_scope(scope_name, reuse=None, initializer=tf.random_normal_initializer()) as scope_model:
#     inputs = tf.placeholder('float32', shape=[None, None, 1], name='inputs')  # [batch, time, in]
#     targets = tf.placeholder('float32', shape=[None, None, 1], name='targets')  # [batch, time, out]
#
#     cell = BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
#     #     cell = DropoutWrapper(cell)
#     init_state = cell.zero_state(1, 'float32')
#
#     rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, inputs, initial_state=init_state, time_major=True)
#     rnn_outputs = tf.reshape(rnn_outputs, [-1, hidden_size])
#
#     w = tf.get_variable('weights', [hidden_size, 1], initializer=tf.random_normal_initializer())
#     b = tf.get_variable('biases', [1, 1], initializer=tf.constant_initializer())
#
#     dense1 = tf.matmul(rnn_outputs, w) + b
#     dense2 = tf.reduce_sum(dense1, reduction_indices=[1])
#     prediction = tf.nn.sigmoid(dense2)
#
#     error2 = tf.square(targets - prediction)
#     error3 = tf.reduce_mean(error2, reduction_indices=[1, 2])
#     train_fn = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error3)
#
#     # Session
#     init_op = tf.global_variables_initializer()
#
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.001, allow_growth=True)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#     sess.run(init_op)


class AirlinePassengerModel(object):
    def __init__(self, size, n_layers, scope_name='airline_passgenger'):
        self.input = tf.placeholder('float32', shape=[None, None, 1], name='input')  # [batch, time, in]
        self.target = tf.placeholder('float32', shape=[None, None, 1], name='targets')  # [batch, time, out]
        self.size = size
        self.n_layers = n_layers
        self.scope_name = scope_name

        with tf.variable_scope(scope_name) as scope:
            # Create RNN Model
            make_rnn = lambda: BasicLSTMCell(size, forget_bias=1.0, state_is_tuple=True)
            self.cell = MultiRNNCell([make_rnn() for _ in range(n_layers)], state_is_tuple=True)
            self._init_state = self.cell.zero_state(size, dtype='float32')

            self.w1 = tf.get_variable('weights', [self.size, self.size], dtype='float32')
            self.b1 = tf.get_variable('bias', [self.size], dtype='float32')

        self._create_session()

    def _create_session(self):
        init_op = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.001, allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.as_default()
        # self.sess.run(init_op)

    def fit(self, train_x, train_y):
        N = max(train_x.get_shape())
        self.N = N

        for epoch in range(2):
            with tf.variable_scope(self.scope_name) as scope:
                if epoch >= 1:
                    scope.reuse_variables()

                output, state, prediction = self.predict(train_x, train_y)

            with tf.variable_scope('train') as scope:
                if epoch >= 1:
                    scope.reuse_variables()

                # Calculate Loss
                loss = tf.reduce_mean(tf.square(train_y - prediction))  # /tf.constant(N, dtype='float32', name='N')
                train_fn = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

            self.final_state = state

            tf.global_variables_initializer().run(session=self.sess)
            # print(self.sess.run(output)[:10])
            # print(train_x[:10])
            print(self.sess.run(train_fn))

    def predict(self, data_x, data_y):
        scope = tf.get_variable_scope()
        outputs = []
        state = self._init_state
        for t in range(self.N):
            cell_output, state = self.cell(data_x[t, :, :], state)
            outputs.append(cell_output)
            if t == 0 and scope:
                scope.reuse_variables()

        output = tf.reshape(tf.concat(outputs, 0), [-1, self.size])
        logits = tf.matmul(output, self.w1) + self.b1
        prediction = tf.nn.sigmoid(logits, name='sigmoid')

        return output, state, prediction

    def make_embedding(self, x, y, name):
        with tf.variable_scope('embedding') as scope:
            embedded_x = tf.get_variable(name + "_x", x.shape, initializer=tf.constant_initializer(x))
            embedded_y = tf.get_variable(name + '_y', y.shape, initializer=tf.constant_initializer(y))
        return embedded_x, embedded_y

    def display(self, tensor):
        self.sess.run(tensor)


model = AirlinePassengerModel(size=1, n_layers=4)
model.fit(*model.make_embedding(train_x, train_y, 'test'))
output, state, prediction = model.predict(*model.make_embedding(test_x, test_y, 'prediction'))
model.display(prediction)


# Train

def train(x, y, test_x, test_y, epoch=2, train_less=0.0019, test_less=0.03):
    global rnn_states

    is_first = True

    now = datetime.now()
    N = len(x)
    g_costs = []
    g_test_costs = []

    for global_step in range(epoch):
        costs = []

        # Shuffle
        permu = np.random.permutation(x.shape[0])
        x = x[permu]
        y = y[permu]

        splitted_x = tf.convert_to_tensor(tf.split(x, [1 for _ in range(N)], axis=0))
        splitted_y = tf.split(y, [1 for _ in range(N)], axis=0)

        cell(splitted_x, rnn_states)
        return

        #             for split_x, split_y in zip(splitted_x, splitted_y):
        #                 print(split_x.eval())
        #                 cost, _ = sess.run([error3, train_fn], feed_dict={inputs: split_x, targets: split_y})
        #                 costs.append(np.sum(cost))

        #                 split_x = tf.reshape(split_x, (1, 5))
        #                 scope.reuse_variables()
        #                 rnn_outputs, rnn_states = cell(split_x, rnn_states)

        outputs = []

        with tf.variable_scope('train') as train_scope:
            if not is_first:
                train_scope.reuse_variables()

            for _ in range(N):
                i = np.random.randint(0, N)
                sample_x = tf.convert_to_tensor(x[i])
                sample_y = tf.convert_to_tensor(y[i])

                rnn_outputs, rnn_states = cell(sample_x, rnn_states, scope=train_scope)
                train_scope.reuse_variables()

                outputs.append(rnn_outputs)

        with tf.variable_scope('train') as train_scope:
            if not is_first:
                train_scope.reuse_variables()

            output = tf.reshape(tf.concat(outputs, 1), [-1, 1])
            logits = tf.matmul(output, w) + b

            sample_y = tf.convert_to_tensor(y.reshape((-1, 1)))
            loss = tf.reduce_mean(tf.square(sample_y - logits), reduction_indices=[1])
            train_fn = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

            sess.run([loss, train_fn], feed_dict={inputs: x})

        is_first = False
        train_scope.reuse_variables()
        print(output)
        print(logits, 'logits', logits.name)
        print(y.reshape((-1, 1)).shape)

        #             loss = tf.square(sample_y - rnn_outputs)
        #             loss = tf.reduce_mean(loss, reduction_indices=[1])
        #             train_fn = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)


        #             cost, _ = sess.run([loss, train_fn], feed_dict={inputs: sample_x, targets: sample_y})


        #                 error2 = tf.square(targets - prediction)
        #                 error3 = tf.reduce_mean(error2, reduction_indices=[1, 2])
        #                 train_fn = tf.train.AdamOptimizer(learning_rate=0.001).minimize(error3)

        #                 cost, _ = sess.run([error3, train_fn], feed_dict={inputs: sample_x, targets: sample_y})
        #                 costs.append(np.sum(cost))

        #                 sample_ = tf.convert_to_tensor(sample_x.reshape(1, 5))
        #                 rnn_outputs, rnn_states = cell(sample_, rnn_states)
        #             break


        # Evaluate Training cost
        _c = np.sum(costs) / len(costs)
        g_costs.append(_c)

        # Evaluate Test Cost
        test_predicted = sess.run(error3, feed_dict={inputs: test_x, targets: test_y})
        test_cost = np.sum(test_predicted) / len(test_predicted)
        g_test_costs.append(test_cost)

        print('\r[%d]Training Cost: %.6f \tTest Cost:%.6f' % (global_step + 1, _c, test_cost), end='')
        if test_less > test_cost and train_less > _c:
            break

    seconds = (datetime.now() - now).total_seconds()
    return seconds, g_costs, g_test_costs

#
#
# seconds, costs, test_costs = train(train_x, train_y, test_x, test_y)
# print('걸린시각:%s 초' % seconds)
