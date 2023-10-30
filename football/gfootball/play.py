import ray
#from football_env import FootballEnv
from gfootball.env import football_env

cfg_values = {
      'action_set': 'full',
      'dump_full_episodes': True,
      'real_time': True,
      'level':"academy_empty_goal"
  }
env =football_env.FootballEnv(cfg_values)

# Load the trained agents.
agent1 = ray.rllib.agents.ppo.PPOTrainer.restore("agent_1.pkl")
agent2 = ray.rllib.agents.ppo.PPOTrainer.restore("agent_2.pkl")

# Add the agents to the environment.
env.add_agent(agent1)
env.add_agent(agent2)

# Start the environment and let the agents play.
env.run()
