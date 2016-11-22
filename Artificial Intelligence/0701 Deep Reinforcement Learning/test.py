import numpy as np

from environment import Environment
from replay import ExperienceReplay
import tensorflow as tf


def test_memory_replay():
    env = Environment('Breakout-v0')
    print 'env.dims:', env.dims
    replay = ExperienceReplay(env)

    env.reset()
    count = 0
    while True:
        count += 1
        action = env.random_action()
        screen, reward, done, info = env.step(action)
        replay.add(screen, reward, action, done)
        if done:
            break

    prestates, actions, rewards, poststates, terminals = replay.sample()

    print 'prestates:', prestates.shape
    print 'actions:', actions
    print 'rewards:', rewards
    print 'poststates:', poststates.shape
    print 'terminals:', terminals
    print 'count:', count

    for i in range(replay.batch_size - 1):
        for j in range(replay.history_size):

            if (j + 1) % replay.history_size != 0:
                print np.array_equal(prestates[i][j + 1], poststates[i][j]), (j + 1) % replay.history_size
            else:
                print np.array_equal(prestates[i][0], poststates[i][j]), (j + 1) % replay.history_size


def test_argmax():
    input = tf.placeholder('float32', [10])
    max = tf.argmax(input, dimension=0)

    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        assert 8 == sess.run(max, feed_dict={input: [10, 20, 30, 40, 50, 60, 70, 80, 90, 0]})
        assert 0 == sess.run(max, feed_dict={input: [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0]})


def test_one_hot_vector():
    input1 = tf.placeholder('int64', [None])

    # One hot vector
    one_hot_tf = tf.one_hot(input1, depth=5, on_value=1., off_value=0.)
    data = [0, 1, 2, 3, 1, 2]
    answer = [[1., 0., 0., 0., 0.],
              [0., 1., 0., 0., 0.],
              [0., 0., 1., 0., 0.],
              [0., 0., 0., 1., 0.],
              [0., 1., 0., 0., 0.],
              [0., 0., 1., 0., 0.]]

    # Reduced Sum
    reduced_sum = tf.reduce_sum(one_hot_tf, reduction_indices=1)
    reduced_sum_answer = [1., 1., 1., 1., 1., 1.]

    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        one_hot = sess.run(one_hot_tf, feed_dict={input1: data})
        assert np.array_equal(answer, one_hot)
        assert np.array_equal(reduced_sum_answer, sess.run(reduced_sum, feed_dict={one_hot_tf: one_hot}))


def test_clip_by_value():
    input1 = tf.placeholder('int64', [None])
    cliped_data = tf.clip_by_value(input1, clip_value_min=2, clip_value_max=5)
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    answer = [2, 2, 3, 4, 5, 5, 5, 5, 5, 5]

    init = tf.initialize_all_variables()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        assert np.array_equal(answer, cliped_data.eval({input1: data}))
