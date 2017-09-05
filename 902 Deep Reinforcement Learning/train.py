import agent
import environment
import replay

env = environment.Environment('Breakout-v0')
replay = replay.ExperienceReplay(env)
agent = agent.Agent(env, replay)
# agent.restore()
agent.train()
