import ray
from ray.rllib.agents import ppo
from gfootball.football_env import FootballEnv

env = FootballEnv(level="academy_empty_goal")
agent = ppo.PPOTrainer(env=env, config={"num_workers": 2})

for i in range(100000):
    agent.train()

agent.save("agent.pkl")
