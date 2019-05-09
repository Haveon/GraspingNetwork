import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rodrigues(axis_angle, v):
    theta = np.linalg.norm(axis_angle)
    rot_vec = axis_angle/theta
    return v*np.cos(theta) + np.cross(rot_vec, v)*np.sin(theta) + rot_vec*np.dot(rot_vec, v)*(1-np.cos(theta))

def scatter_series(x_data, y_data):
    plt.scatter(x_data, y_data)
    plt.xlabel(x_data.name)
    plt.ylabel(y_data.name)

def quick_viz(frame):
    plt.figure(figsize=(16,9), dpi=100)
    plt.subplot(321)
    scatter_series(frame.x, frame.y)
    plt.subplot(322)
    scatter_series(frame.x, frame.z)
    plt.subplot(323)
    scatter_series(frame.y, frame.z)
    plt.subplot(313)
    plt.hist(np.linalg.norm(frame[['x', 'y', 'z']].values, axis=-1), bins=300)
    plt.xlabel('Distance from table center')

def rot_viz(frame):
    fig = plt.figure(figsize=(16,9), dpi=100)
    ax = fig.add_subplot(211, projection='3d')
    ax.scatter(frame.rx, frame.ry, frame.rz, s=50)
    ax.set_aspect(1)
    ax = fig.add_subplot(212)
    ax.hist(frame.rot_mag, bins=500)

def move_grasp_closer(data, length, visualize=False):
    """
    data should have x,y,z,rx,ry,rz columns. Changes data in place
    length is float and should correspond to how close to move the grasp
    along the gripper's z-axis
    """
    rot = data[['rx', 'ry', 'rz']].values
    pointing = np.zeros(rot.shape)
    for i in range(rot.shape[0]):
        pointing[i] = rodrigues(rot[i], np.array([0,0,1]))

    # Move length distance in palm direction
    comp = (data[['x', 'y', 'z']] + length*pointing)

    if visualize:
        quick_viz(data)
        plt.suptitle('Before compression')
        quick_viz(comp)
        plt.suptitle('After compression')

    data.x = comp.x
    data.y = comp.y
    data.z = comp.z
    return

def limit_to_cube(data, xlim=[-0.2,0.2], ylim=[-0.2,0.2], zlim=[0.05,0.5], visualize=False):
    """
    Returns a new data frame with only points that lie inside the cube

    xlim,ylim,zlim should be set as [low,high], that is we keep things higher than
    "low" but less than "high"
    """
    x_bool = (data.x >= xlim[0]) & (data.x <= xlim[1])
    y_bool = (data.y >= ylim[0]) & (data.y <= ylim[1])
    z_bool = (data.z >= zlim[0]) & (data.z <= zlim[1])

    data = data[x_bool & y_bool & z_bool]

    if visualize:
        quick_viz(data)
        plt.suptitle('Keep only above table')
    return data

def drop_outliers(data, total_drop, visualize=False):
    """
    Returns a new data frame without the outliers

    total_drop should be a multiple of 6 for best results
    """
    per_axis_drop = total_drop//3
    keep_rows = data.z > 0 # Need a series of all True values
    for axis in ['x', 'y', 'z']:
        top_axis = data.nlargest(n=per_axis_drop//2, columns=axis).iloc[-1]
        bot_axis = data.nsmallest(n=per_axis_drop//2, columns=axis).iloc[-1]
        keep_axis = (data[axis]>bot_axis[axis]) & (data[axis]<top_axis[axis])
        keep_rows = keep_rows & keep_axis

    data = data[keep_rows]

    if visualize:
        quick_viz(data)
        plt.suptitle('Drop 100 in each direction')
    return data

def extract_axis_angle(data, visualize=False):
    """
    Changes data in place by normalizing rx,ry,rz and adding a rot_mag column
    """
    rot_mag = np.linalg.norm(data[['rx', 'ry', 'rz']].values, axis=-1)
    data.rx = data.rx/rot_mag
    data.ry = data.ry/rot_mag
    data.rz = data.rz/rot_mag
    rot_mag = rot_mag % (2*np.pi)
    data['rot_mag'] = rot_mag
    if visualize:
        rot_viz(data)
    return

def make_quaternion_from_axis_angle(axis, angle):
    """
    Angle must be in radians
    Axis must have norm 1
    """
    q0 = np.cos(angle/2)
    q = axis*np.sin(angle/2)
    return np.array([q0,q[0],q[1],q[2]])

def multiply_quaternions(q, p):
    """
    Both quaternions must be (4,) shaped ndarrays with the first element being
    the scalar component followed by i,j,k in that order.
    """
    scalar = q[0]*p[0] - np.dot(q[1:], p[1:])
    vector = q[0]*p[1:] + p[0]*q[1:] + np.cross(q[1:], p[1:])
    return np.array([scalar, vector[0], vector[1], vector[2]])

def make_axis_angle_from_quaternion(q):
    """
    q needs to be a rotation quaternion
    """
    norm = np.linalg.norm(q[1:])
    angle = 2*np.arctan2(norm, q[0])
    if norm==0:
        axis = np.zeros(3)
    else:
        axis = q[1:]/norm
    return axis, angle

def rotate_vector(v, q):
    v_quat = np.zeros(4)
    v_quat[1:] = v
    q_star = q*np.array([1,-1,-1,-1])
    return multiply_quaternions(q, multiply_quaternions(v_quat, q_star))[1:]

def rotate_to_camera_coordinates(data, visualize=False):
    """
    Changes data in place to camera coordinates from table coordinates
    """

    rot1 = make_quaternion_from_axis_angle(np.array([1,0,0]), np.deg2rad(-102))
    rot2 = make_quaternion_from_axis_angle(np.array([0,1,0]), np.deg2rad(-0.9))
    rot3 = make_quaternion_from_axis_angle(np.array([0,0,1]), np.deg2rad(-135))
    table_to_camera = multiply_quaternions(rot3, multiply_quaternions(rot2,rot1))*np.array([1,-1,-1,-1])
    vec_to_cam = np.array([-0.353906, 0.286732, 0.2275])

    position = data[['x','y','z']].values.copy()
    rotation = data[['rx','ry','rz']].values.copy()
    rotation_magnitude = data.rot_mag.values.copy()

    for i in range(position.shape[0]):
        pos = position[i]
        rot = rotation[i]
        rot_mag = rotation_magnitude[i]

        new_pos = rotate_vector(pos - vec_to_cam, table_to_camera)
        q = make_quaternion_from_axis_angle(rot, rot_mag)
        new_q = multiply_quaternions(table_to_camera, q)
        new_rot, new_rot_mag = make_axis_angle_from_quaternion(new_q)

        position[i] = new_pos
        rotation[i] = new_rot
        rotation_magnitude[i] = new_rot_mag

    data[['x', 'y', 'z']] = position
    data[['rx','ry','rz']]= rotation
    data.rot_mag = rotation_magnitude

    if visualize:
        quick_viz(data)
        plt.suptitle('In Camera Coordinates')
    return

def assign_voxels_to_position(data, voxel_centers):
    position = data[['id', 'x', 'y', 'z']].values
    cube = np.zeros([data.shape[0],4], dtype=np.int32)
    for i in range(position.shape[0]):
        ind = np.argmin(np.linalg.norm(voxel_centers - position[i,1:], axis=-1))
        # Voxel values are set as X,Y,Z
        voxel = np.unravel_index(ind, (8,8,8))
        cube[i, 0] = position[i,0]
        cube[i,1:] = voxel
    return cube

# ----------------------------------------------------------------------------

# positive data is in millimeters
positive_data = pd.read_csv('raw_data/gripper_pose (rev 1 - closest center).tsv', sep='\t')

# Conver to meters
positive_data.x = positive_data.x/1000
positive_data.y = positive_data.y/1000
positive_data.z = positive_data.z/1000

# negative data is in meters
negative_data = pd.read_csv('raw_data/negative_grasp_data.csv')

# Move 5cm closer along gripper z-axis
move_grasp_closer(positive_data, 0.05)
move_grasp_closer(negative_data, 0.05)

positive_data = limit_to_cube(positive_data, xlim=[-0.2,0.2], ylim=[-0.2,0.2], zlim=[0.05,0.5])
positive_data = drop_outliers(positive_data, total_drop=600)

# Find matching ids between positive and negative data
ids = list(set(negative_data.id.values).intersection(set(positive_data.id.values)))
columns_to_keep = ['id','x','y','z','rx','ry','rz']
positive_data = positive_data[positive_data.id.isin(ids)][columns_to_keep]
negative_data = negative_data[negative_data.id.isin(ids)]

extract_axis_angle(positive_data)
extract_axis_angle(negative_data)

positive_data = positive_data.sort_values(by='id')
negative_data = negative_data.sort_values(by='id')

positive_data.to_csv('matched_up_data/2percent_removed/matched_pos_data_in_table_coordinates.csv', index=False)
negative_data.to_csv('matched_up_data/2percent_removed/matched_neg_data_in_table_coordinates.csv', index=False)

rotate_to_camera_coordinates(positive_data)
rotate_to_camera_coordinates(negative_data)

positive_data.to_csv('matched_up_data/2percent_removed/matched_pos_data_in_camera_coordinates.csv', index=False)
negative_data.to_csv('matched_up_data/2percent_removed/matched_neg_data_in_camera_coordinates.csv', index=False)

dx = (positive_data.x.max() - positive_data.x.min())/8
dy = (positive_data.y.max() - positive_data.y.min())/8
dz = (positive_data.z.max() - positive_data.z.min())/8
xs = [positive_data.x.min() + dx/2 + i*dx for i in range(8)]
ys = [positive_data.y.min() + dy/2 + i*dy for i in range(8)]
zs = [positive_data.z.min() + dz/2 + i*dz for i in range(8)]
X,Y,Z = np.meshgrid(xs,ys,zs, indexing='ij')
voxel_centers = np.stack([X,Y,Z], axis=-1)

positive_voxel_indeces = assign_voxels_to_position(positive_data, voxel_centers)
negative_voxel_indeces = assign_voxels_to_position(negative_data, voxel_centers)

np.savetxt('cubes/2percent_removed/positive_position_cube_compact.csv',
            positive_voxel_indeces, fmt='%d', delimiter=',', header='id,x,y,z')
np.savetxt('cubes/2percent_removed/negative_position_cube_compact.csv',
            negative_voxel_indeces, fmt='%d', delimiter=',', header='id,x,y,z')
np.savetxt('cubes/2percent_removed/voxel_centers.csv', voxel_centers.reshape([512,3]),
            delimiter=',', header='x,y,z')
