import gymnasium as gym
import torch as th
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from ocean_env import OceanEnvironment
import holoocean
import time

# Create the Ocean environment
env = OceanEnvironment(holoocean.make("PierHarbor-Hovering"))

# Create a PPO agent with the custom policy
# policy_kwargs={'net_arch':              dict(pi=[64, 64], vf=[64, 64]),
#                'activation_functions':  [th.nn.ReLU, th.nn.Tanh]}
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=dict(pi=[32, 32], vf=[32, 32])
                    )
model = PPO(ActorCriticPolicy, env, verbose=1, policy_kwargs = policy_kwargs, seed=123, use_sde=True)


start_time = time.time()
# Train the agent
model.learn(total_timesteps=1000)  # Train for 10 time steps
end_time = time.time()
print("running time(mins): ",(end_time-start_time)/60)
# Save the trained model
model.save("ppo_cartpole_custom")

# Close the environment
env.close()
