import numpy as np

from environment import Environment
from replay import ExperienceReplay


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

    for i in range(replay.batch_size -1):
        for j in range(replay.history_size):

            print i, j, ':',
            if (j + 1) % replay.history_size != 0:
                print np.array_equal(prestates[i][j + 1], poststates[i][j]), (j + 1) % replay.history_size
            else:
                print np.array_equal(prestates[i+1][0], poststates[i][j]), (j + 1) % replay.history_size

        print
