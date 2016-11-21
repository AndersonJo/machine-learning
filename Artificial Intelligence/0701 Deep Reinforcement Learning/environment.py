import gym
import cv2


class Environment(object):
    def __init__(self, game):
        self.game = gym.make(game)
        self.action_size = self.game.action_space.n
        self.height, self.width, _ = self.shape = self.game.observation_space.shape

    def play_sample(self):
        while True:
            self.game.render()
            action = self.game.action_space.sample()
            observation, reward, done = self.step(action)
            if done:
                break
        self.game.close()

    def step(self, action, gray=True):
        observation, reward, done, info = self.game.step(action)
        if gray:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation, reward, done

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
