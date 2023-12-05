import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def generate_cuboid(center, size):
    # Given the center and size, generate the vertices of a cuboid
    half_size = size[0] / 2.0
    vertices = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0],
        [-half_size, -half_size, size[2]],
        [half_size, -half_size, size[2]],
        [half_size, half_size, size[2]],
        [-half_size, half_size, size[2]]
    ]) + center
    return vertices

def generate_cuboid_in_plot(ax_3d, num_cuboids = 3):
    # Generate random positions for the cuboids
    positions = np.random.uniform(low=(-50, -50, 0), high=(50, 50, 0), size=(num_cuboids, 3))
    # Fixed size for all cuboids
    cuboid_size = np.array([20, 20, 1])
    # Plot colors
    cuboids_color = np.zeros((num_cuboids,3))
    cuboids_color[num_cuboids-1][0] = 1

    # all cuboid upper faces
    upper_faces_of_all_cuboid = []

    # Plot each cuboid
    for i in range(num_cuboids):
        cuboid_vertices = generate_cuboid(positions[i], cuboid_size)
        cuboid_upper_face = [[cuboid_vertices[4], cuboid_vertices[5], cuboid_vertices[6], cuboid_vertices[7]]]
        upper_faces_of_all_cuboid.append(np.concatenate(cuboid_upper_face, axis=0))
        cuboid_collection = Poly3DCollection(cuboid_upper_face, color=cuboids_color[i])
        ax_3d.add_collection3d(cuboid_collection)

    # Set plot limits
    ax_3d.set_xlim([-50, 50])
    ax_3d.set_ylim([-50, 50])
    ax_3d.set_zlim([0, 20])

    # Set axis labels
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')

    return ax_3d, upper_faces_of_all_cuboid

def  add_flight_to_plot(ax_3d, flight_pose, prev_flight_pose):
    prev_pos = prev_flight_pose[0:3]
    pos = flight_pose[0:3]
    robot1 = ax_3d.scatter(prev_pos[0],prev_pos[1], prev_pos[2], color='g', marker='*', s=100, label='prev_pose')
    robot2 = ax_3d.scatter(pos[0],pos[1],pos[2], color='r', marker='*', s=100, label='cur_pose')
    return robot1, robot2
