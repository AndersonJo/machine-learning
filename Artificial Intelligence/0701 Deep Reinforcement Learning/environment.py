import gym
import cv2


class Environment(object):
    def __init__(self, game):
        self.game = gym.make(game)
        self.action_size = self.game.action_space.n
        self.height, self.width = self.dims = (84, 84)

    def play_sample(self):
        while True:
            self.game.render()
            action = self.game.action_space.sample()
            observation, reward, done = self.step(action)
            if done:
                break
        self.game.close()

    def step(self, action, gray=True, resize=True):
        screen, reward, done, info = self.game.step(action)
        screen = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY), (self.height, self.width))
        return (screen, reward, done, info)

    def random_step(self, gray=True):
        action = self.game.action_space.sample()
        return self.step(action, gray=True)

    def random_action(self):
        return self.game.action_space.sample()

    def render(self):
        return self.game.render()

    def reset(self):
        return self.game.reset()

    def close(self):
        return self.game.close()
