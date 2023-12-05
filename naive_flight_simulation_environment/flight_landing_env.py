import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
sys.path.append("/home/shen/UM-MS-study/FROG-lab")
import transforms3d.euler as euler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from generate_platforms import generate_cuboid_in_plot, add_flight_to_plot
from capture_picture_from_mono_camera import picture_from_camera

class FlightLandingEnv(gym.Env):
    def __init__(self,mode = 'rgb_array', max_steps = 100):
        super(FlightLandingEnv, self).__init__()

        # Define the pose of the flight
        x_range = (-50.0, 50.0)    # Range for x
        y_range = (-50.0, 50.0)    # Range for y
        z_range = (0.0, 100.0)     # Range for z
        roll_range = (0.0, 0.0)   # Range for roll
        pitch_range = (0.0, 0.0)  # Range for pitch
        yaw_range = (-np.pi, np.pi)    # Range for yaw
        image_size_x = 800
        image_size_y = 500
        # Initialize platform numbers. One of the platforms will be set to red while others will be black

        # Create a list of bounds for each dimension
        bounds = [x_range, y_range, z_range, roll_range, pitch_range, yaw_range]
        bounds_action = [x_range, y_range, (-10.0, 10.0), roll_range, pitch_range, yaw_range]

        # Define the observation space
        self.observation_space = spaces.Dict({
            'pose':  spaces.Box(low=np.array([bound[0] for bound in bounds]), high=np.array([bound[1] for bound in bounds]), dtype=float),  # Flight pose)
            'image': spaces.Box(low=0, high=255, shape=(image_size_x, image_size_y, 3), dtype=np.uint8)  # RGB image
        })

        # Define the action space
        self.action_space = spaces.Box(low=np.array([bound[0] for bound in bounds_action]), high=np.array([bound[1] for bound in bounds_action]), dtype=float) # Continuous 3D action space

        # set render mode
        self.render_mode = mode

        # set max steps
        self.max_steps = max_steps

        #Initialize everything else
        self.reset_times = 0
        self.reset()



    def step(self, action):
        # Move the flight based on the chosen action
        self.prev_flight_pose = np.copy( self.flight_pose)
        self.flight_pose = np.copy(action)
        # offset of z so x y z is normalized
        self.flight_pose[2] += 10.0
        self.camera_position = np.copy( self.flight_pose)
        self.camera_position[3] += np.pi

        # Check if the flight is over a red platform
        dist_from_red = np.linalg.norm(self.flight_pose[0:2]-self.red_platform_center[0:2])
        is_reach = (dist_from_red < 10 and (self.flight_pose[2]-self.red_platform_center[2] < 10))
        terminated = (is_reach == True)
        # out of boundary
        truncated = (self.action_space.contains(action)==False) or self.count >= self.max_steps

        # Reward: -1 for each step, additional reward for landing on the red platform
        reward = -1+10.0/dist_from_red if (is_reach == False) else 100
        if(reward > 50):
            print("reward: ",reward)
        # print("current steps: ", self.count)

        # Observation: Flight pose and RGB image
        self.count += 1
        observation = self.get_obs(show=True)
        

        return observation, reward, terminated, truncated , {}

    def reset(self, seed=None, options=None):
        # Reset flight pose and camera positions
        self.flight_pose = np.array([0.0, 0.0, 60.0, 0.0, 0.0, 0.0])
        self.prev_flight_pose = np.copy(self.flight_pose)
        self.prev_flight_pose[2] += 10.0
        self.camera_position = self.flight_pose.copy()
        self.camera_position[3] += np.pi # camera facing down

        if hasattr(self,"fig_human_view"):
            # plt.close(self.fig_human_view)
            plt.close()
            self.ax_3d = None
            self.ax_2d = None

        # Figures in human view or camera(robot) view
        self.fig_human_view = plt.figure(1)
        ax_3d = self.fig_human_view.add_subplot(121, projection='3d')
        self.ax_3d, self.upper_faces_of_all_cuboid = generate_cuboid_in_plot(ax_3d,3)
        self.ax_2d = self.fig_human_view.add_subplot(122)

        # if(self.render_mode != 'human'):
        #     # No need to render
        #     self.fig_human_view = None
        #     self.ax_3d = None
        #     self.ax_2d = None

        # Reset remaining steps
        # self.max_steps = 100
        self.count = 0

        # red platform place
        center_x = (self.upper_faces_of_all_cuboid[-1][0][0]+self.upper_faces_of_all_cuboid[-1][1][0])/2.0
        center_y = (self.upper_faces_of_all_cuboid[-1][0][1]+self.upper_faces_of_all_cuboid[-1][2][1])/2.0
        center_z = self.upper_faces_of_all_cuboid[-1][0][2]
        self.red_platform_center = np.array([center_x, center_y, center_z])

        # Return initial observation and info
        info = None

        self.reset_times += 1
        print("reset time: ", self.reset_times)
        return self.get_obs(), info

    def render(self, show):
        # Render the environment with the flight and camera positions
        image = picture_from_camera(self.upper_faces_of_all_cuboid, self.camera_position)
        if self.render_mode == 'human' and show==True:
            robot_pos_plot1,robot_pos_plot2 = add_flight_to_plot(self.ax_3d, self.flight_pose, self.prev_flight_pose)
            # print(self.count)
            camera_image = self.ax_2d.imshow(image)
            plt.pause(2)
            # plt.show()
            camera_image.remove()
            robot_pos_plot1.remove()
            robot_pos_plot2.remove()
        return image



    def get_obs(self, show = False):
        return {
            'pose': np.copy(self.flight_pose),
            'image': self.render(show)
        }