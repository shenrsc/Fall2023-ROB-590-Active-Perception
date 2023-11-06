import numpy as np
import transforms3d.euler as euler



def Clamp(rad:float):
    while(rad<-np.pi):
        rad+=2*np.pi
    while(rad>=np.pi):
        rad-=2*np.pi
    return rad

def TimeStepPDController(cur_pose_matrix, prev_error, goal_pose, T_period):
    # Input:
    # cur_pose_matrix:     current pose of this robot, a 4x4 homogonous matrix
    # prev_error:   previous error, a 1x6 or 6x1 vector
    # goal pose:    is a 1x6 or 6x1 vector. represents x,y,z,row,pitch, yaw
    # T_period:      a time period of a command

    # output: 
    # command:       a 1x8 vector, command output for PierHarbor
    # prev_error：      previous error for next time step, a 1x6 or 6x1 vector
    
    Kp = 400.0
    Kd = 1000.0

    goal_xyz = goal_pose[0:3]
    goal_row = goal_pose[3]
    goal_pitch = goal_pose[4]
    goal_yaw = goal_pose[5]
    rotation_matrix = cur_pose_matrix[:3, :3]
    roll, pitch, yaw = euler.mat2euler(rotation_matrix, axes='sxyz')
    translation_vector = cur_pose_matrix[:3,3]

    cur_error = np.zeros(6)
    cur_error[0:3] = np.ravel(goal_xyz) - translation_vector
    cur_error[0:3] = np.dot(euler.euler2mat(Clamp(roll+np.pi), pitch, -yaw, axes='sxyz'),cur_error[0:3])
    cur_error[3:6] = np.array([Clamp(goal_row-roll), Clamp(goal_pitch-pitch), Clamp(goal_yaw-yaw)])
    error_diff = (cur_error-prev_error)/T_period

    u_output = Kp*cur_error + Kd*error_diff
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

    return command, prev_error

def PDcontroller(env, goal_pose:np.array, control_rate:int=20, pose_err_endurance:float=0.3, max_action_time:float=100.0):
    # Input:
    # env:                  simulation environment
    # goal pose:            is a 1x6 or 6x1 vector. represents x,y,z,row,pitch, yaw
    # control_rate:         how many control commands should be generated per second
    #pose_err_endurance:    how much error we allow for pose
    #max_action_time:       How long we can have for the action to gogal_pose. If the robot can not arrive within this time, simulation falls
    
    #the z axis of world is different from z axis of 
    goal_pose[3] =Clamp(goal_pose[3]-np.pi)
    cur_pose = goal_pose - 2.0 #init cur_pose so it will be different from goal_pose at the beginning
    prev_controller_error = np.zeros(6)
    T_period = 1.0/control_rate
    time = 0.0
    while(np.linalg.norm(cur_pose-goal_pose) > pose_err_endurance and (time < max_action_time) ):
        time += T_period
        state = env.tick()
        if 'PoseSensor' in state:
            cur_pose_matrix = state['PoseSensor'] # a homogonous matrix
            # Extract the rotation matrix (3x3) from the homogeneous matrix
            rotation_matrix = cur_pose_matrix[:3, :3]
            # Use the rotation matrix to calculate roll, pitch, and yaw angles
            roll, pitch, yaw = euler.mat2euler(rotation_matrix, axes='sxyz')
            translation_vector = cur_pose_matrix[:3,3]
            cur_pose = np.append(translation_vector,[roll, pitch, yaw])
            controller_command, prev_controller_error = TimeStepPDController(cur_pose_matrix, prev_controller_error, goal_pose, T_period)
            state = env.step(controller_command)

        else:
            print("ERROR! No pose of robotics detected!")
            continue
    
    return time #time is a cost
