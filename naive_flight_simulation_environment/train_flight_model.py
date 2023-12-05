import gymnasium as gym
import torch as th
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from flight_landing_env import FlightLandingEnv
from  flight_feature_extractor import FlightFeatureExtractor
import time

# Create the naive simulation environment
env = FlightLandingEnv(mode = 'rgb_array',max_steps=10)

# Create a PPO agent with the custom policy
# policy_kwargs={'net_arch':              dict(pi=[64, 64], vf=[64, 64]),
#                'activation_functions':  [th.nn.ReLU, th.nn.Tanh]}
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                    net_arch=dict(pi=[64, 64], vf=[64, 64]),
                    features_extractor_class = FlightFeatureExtractor
                    )
model = PPO(ActorCriticPolicy, env, verbose=1, policy_kwargs = policy_kwargs, seed=123, use_sde=True)


start_time = time.time()
# Train the agent
model.learn(total_timesteps=100)  # Train for 10 time steps
end_time = time.time()
print("running time(mins): ",(end_time-start_time)/60)
# Save the trained model
model.save("ppo_flight_env_custom")

# Close the environment
env.close()
