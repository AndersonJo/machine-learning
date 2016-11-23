import numpy as np

import tensorflow as tf
import tflearn


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
