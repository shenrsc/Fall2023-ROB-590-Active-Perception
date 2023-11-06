import holoocean
import numpy as np
import transforms3d.euler as euler
import matplotlib.pyplot as plt


def clamp(rad:float):
    while(rad<-np.pi):
        rad+=2*np.pi
    while(rad>=np.pi):
        rad-=2*np.pi
    return rad

env = holoocean.make("PierHarbor-Hovering")

Kp = 400.0
Ki = 0.0
Kd = 1000.0

# A trajectory to follow, the trajectory is a circle, r = 10m
T_steps  = 0.1 #time step as 0.01 s
T_total = 100 #100s in total to finish trajectory
r = 10.0

trajectory_t = np.linspace(0,T_total, int(T_total/T_steps)+1)
trajectory_yaw = -np.pi/2*np.ones(trajectory_t.shape[0])#trajectory_t/T_total*2.0*np.pi #have two circules in 100s
# trajectory_yaw = 0.0*trajectory_t
trajectory_pitch = np.zeros(trajectory_t.shape[0])
trajectory_row = np.zeros(trajectory_t.shape[0])
# trajectory_x = r*np.cos(trajectory_yaw)+486.0-r
trajectory_x = np.zeros(trajectory_t.shape[0])+495#486.0
# trajectory_y = r*np.sin(trajectory_yaw)-632.0
trajectory_y = np.zeros(trajectory_t.shape[0])-640.0#632.0
trajectory_z = np.zeros(trajectory_t.shape[0])-10.0
trajectory_xyz = np.stack((trajectory_x, trajectory_y, trajectory_z))



prev_error = np.zeros(6) # error for (x,y,z,row,pitch,yaw)
error_integral = np.zeros(6)
state = env.tick()
whole_error_matrix = np.zeros((6,trajectory_t.shape[0]))
pose_matrix = np.zeros((6,trajectory_t.shape[0]))

for i in range(trajectory_t.shape[0]):
    if 'PoseSensor' in state:
        cur_pose = state['PoseSensor'] # a homogonous matrix
        # Extract the rotation matrix (3x3) from the homogeneous matrix
        rotation_matrix = cur_pose[:3, :3]
        # Use the rotation matrix to calculate roll, pitch, and yaw angles
        roll, pitch, yaw = euler.mat2euler(rotation_matrix, axes='sxyz')
        
        translation_vector = cur_pose[:3,3]

        cur_error = np.zeros(6)
        cur_error[0:3] = np.ravel(trajectory_xyz[:,i]) - translation_vector
        cur_error[0:3] = np.dot(euler.euler2mat(clamp(roll+np.pi), pitch, -yaw, axes='sxyz'),cur_error[0:3])
        cur_error[3:6] = np.array([clamp(trajectory_row[i]-roll), clamp(trajectory_pitch[i]-pitch), clamp(trajectory_yaw[i]-yaw)])
        whole_error_matrix[:,i] = cur_error
        error_integral += cur_error
        error_diff = (cur_error-prev_error)/T_steps
        pose_matrix[0:3,i] = translation_vector
        pose_matrix[3:6,i] = [roll, pitch, yaw]

        u_output = Kp*cur_error + Ki*error_integral + Kd*error_diff
        command = np.zeros(8)
        command[0:4] += u_output[2]/4.0 # thrust for z
        command[4:8] += u_output[0]/4.0 # thrust for x
        # thrust for y
        command[[4,6]] += u_output[1]/4.0 
        command[[5,7]] -= u_output[1]/4.0
        # thrust for yaw
        command[[4,7]] +=  u_output[5]/4.0 
        command[[5,6]] -=  u_output[5]/4.0 
        """currently, I do not know which motor to control pitch and row, so necglect it"""

        prev_error = cur_error
        # if(i==0):
        #     print()
        if(i<20):
            print("rotation ", np.array([roll, pitch, yaw]))
            print("translation vector", translation_vector)
            print("error ", cur_error)
            print("u_output: ",u_output)
            print("command: ",command)
        state = env.step(command)

    else:
        continue


# print(env._get_full_state())
# print(env._get_single_state())
# for i in range(200):
#     command = np.zeros(8)
#     state = env.step(command)
#     # print(state)


if 'PoseSensor' in state:
    cur_pose = state['PoseSensor'] # a homogonous matrix
    roll, pitch, yaw = euler.mat2euler(rotation_matrix, axes='sxyz')
    translation_vector = cur_pose[:3,3]
    print("rotation ", np.array([roll, pitch, yaw]))
    print("translation vector", translation_vector)

# # plot
# fig, ax = plt.subplots()
# ax.plot(trajectory_t, pose_matrix[5,:] , linewidth=2.0)
# plt.show()

        
