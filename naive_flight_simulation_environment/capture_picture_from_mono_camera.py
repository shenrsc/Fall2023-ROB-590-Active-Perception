import numpy as np
import cv2
import matplotlib.pyplot as plt
import transforms3d.euler as euler
from typing import Union
from generate_platforms import generate_cuboid_in_plot, add_flight_to_plot

def picture_from_camera(upper_faces_of_all_cuboid:list, camera_pose:np.array, camerea_setting: Union[None, dict] = None):
    # Input:
    # upper_faces_of_all_cuboid:     a list of np.arrays which represents cuboid. each element is a 4x3 array which represents 3D posistion of 4 vertices of a rectangle surface 
    # camera_pose:                   a 1x6 vector ,the pose of the camera in world coordinate
    # camerea_setting:               camera settings, includes focal_length, principal pointx, principal pointy in order to form intrinsic matrix

    # output: 
    # image      The image created by the camera in simulation

    if(camerea_setting == None):
        focal_length_x = 7.024808718760045
        focal_length_y = 7.003821494715315
        principal_point_x = 400
        principal_point_y = 250
        image_size_x = 800
        image_size_y = 500
    else:
        focal_length_x = camerea_setting["focal_length_x"]
        focal_length_y = camerea_setting["focal_length_y"]
        principal_point_x = camerea_setting["principal_point_x"]
        principal_point_y = camerea_setting["principal_point_y"]
        image_size_x = camerea_setting["image_size_x"]
        image_size_y = camerea_setting["image_size_y"]

    # Camera parameters
    K = np.array([[focal_length_x,    0,              principal_point_x],
                [0,               focal_length_y,   principal_point_y],
                [0,               0,              1]])

    # Camera pose (extrinsics)
    R = euler.euler2mat(camera_pose[3],camera_pose[4], camera_pose[5], axes='sxyz')
    t = np.array(camera_pose[0:3])

    # 3D coordinates of cuboid corners
    cuboid_corners_3d = np.concatenate(upper_faces_of_all_cuboid, axis=0)

    # Project 3D points to 2D image coordinates
    cuboid_corners_2d = cv2.projectPoints(cuboid_corners_3d, R, t, K, None)[0].squeeze()

    # Create an image
    image_size = (image_size_x, image_size_y)
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8)*255

    # Draw squares on the image
    for i in range(0, len(cuboid_corners_2d), 4):  # Assumes 4 corners per cuboid
        square_points = cuboid_corners_2d[i:i+4].astype(int)
        ##fill the area inside the outlier 
        if(i==len(cuboid_corners_2d)-4):
            # The sepcial square is red
            color = (255,0,0)
        else:
            color =  (0,0,0)
        cv2.fillPoly(image, [square_points], color=color)
    return image





