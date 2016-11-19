import gym


class Environment(object):
    def __init__(self, game):
        self.env = gym.make(game)

    def play(self):
        env = self.env

        for i in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if reward:
                print observation, reward, info

            if done:
                break





if __name__ == '__main__':
    env = Environment('Breakout-v0')
    env.play()
