import gymnasium as gym
import numpy as np
import sys
sys.path.append("/home/shen/UM-MS-study/FROG-lab")
from Holo_simulation.PD_position_controller.controller import PDcontroller
from Holo_simulation.PD_position_controller.controller import Clamp
import transforms3d.euler as euler

class OceanEnvironment(gym.Env):
    def __init__(self,holoocean_env):
        super(OceanEnvironment, self).__init__()

        # Define your custom environment parameters and variables here
        # Define the ranges (bounds) for each dimension
        x_range = (-10.0, 10.0)    # Range for x
        y_range = (-10.0, 10.0)    # Range for y
        z_range = (-3.0, 3.0)     # Range for z
        roll_range = (0.0, 0.0)   # Range for roll
        pitch_range = (0.0, 0.0)  # Range for pitch
        yaw_range = (-np.pi/2, np.pi/2)    # Range for yaw

        # Create a list of bounds for each dimension
        bounds = [x_range, y_range, z_range, roll_range, pitch_range, yaw_range]

        # Define the continuous action space using the bounds
        self.action_space = gym.spaces.Box(low=np.array([bound[0] for bound in bounds]), high=np.array([bound[1] for bound in bounds]), dtype=float)
        
        #currently set
        self.observation_space = gym.spaces.Box(low=np.array([bound[0] for bound in bounds]), high=np.array([bound[1] for bound in bounds]), dtype=float)
        
        self.holo_env = holoocean_env  # Initialize your holo_env
        self.evn_start_state = self.get_obs_before_normalize() #get unnormalized original state info
        self.steps_count = 0

    def reset(self, seed=None, options=None):
        # Reset the environment to its initial holo_env
        super().reset(seed=seed)
        self.steps_count = 0
        self.holo_env.reset()
        info = self.get_info()  # Additional information, if needed
        observation = self.get_obs()
        return observation, info

    def step(self, action):
        # Take a step in the environment
        # calculate reward and update holo_env at the same time
        self.steps_count += 1
        print("action:\n",action)
        reward = -PDcontroller(self.holo_env, action+self.evn_start_state)
        print(reward)
        done = self.is_terminal()
        info = self.get_info()  # Additional information, if needed
        observation = self.get_obs()
        unnormalized_obs = self.get_obs_before_normalize()
        print("observed state after action:\n",observation)
        print("unnormalized_obs:\n",unnormalized_obs)
        return observation, reward, done, False, info

    def render(self, mode='human'):
        # implement a rendering method for visualization
        # skip currently
        pass

    def close(self):
        # Clean up resources, if needed
        pass

    # Implement any custom methods for your environment
    def compute_reward(self, holo_env, action):
        # Define your reward computation logic
        return 0

    def transition(self, holo_env, action):
        # Define your holo_env transition logic
        #  action is the goal pose
        return action

    def is_terminal(self):
        # Define your terminal holo_env logic
        return (self.steps_count > 30)
    
    def get_info(self):
        return self.holo_env._get_single_state()
    
    def get_obs(self):
        observation = self.get_obs_before_normalize()
        observation -= self.evn_start_state
        return observation
    
    def get_obs_before_normalize(self):
        info = self.get_info()
        if 'PoseSensor' in info:
            cur_pose_matrix = info['PoseSensor'] # a homogonous matrix
            # Extract the rotation matrix (3x3) from the homogeneous matrix
            rotation_matrix = cur_pose_matrix[:3, :3]
            # Use the rotation matrix to calculate roll, pitch, and yaw angles
            roll, pitch, yaw = euler.mat2euler(rotation_matrix, axes='sxyz')
            translation_vector = cur_pose_matrix[:3,3]
            observation = np.append(translation_vector,[Clamp(roll+np.pi), pitch, yaw])
        else:
            observation = np.zeros((6,1))
        return observation

