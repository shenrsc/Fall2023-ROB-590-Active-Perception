import holoocean
import numpy as np
import transforms3d.euler as euler
from controller import PDcontroller


env = holoocean.make("PierHarbor-Hovering")
env.agents['auv0'].set_control_scheme(3)
# env.set_control_scheme('auv0',3)
aa = env.agents['auv0'].act(np.array([495,-640,-10.0]))
env.tick()
# goal_pose = np.array([495.0, -640.0, -10.0, 0, 0.0, -np.pi/2])
# PDcontroller(env, goal_pose)
env.reset()













































