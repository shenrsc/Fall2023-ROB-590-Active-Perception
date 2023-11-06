import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from ocean_env import OceanEnvironment
from stable_baselines3.common.evaluation import evaluate_policy
import holoocean

env = OceanEnvironment(holoocean.make("PierHarbor-Hovering"))
model = PPO.load("ppo_cartpole_custom", env=env)

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)