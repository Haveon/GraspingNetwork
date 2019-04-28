import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

x_lim = abs(positive_data.x)<0.2
y_lim = abs(positive_data.y)<0.2
z_lim = (positive_data.z>0.05) & (positive_data.z<0.5)

positive_data = positive_data[x_lim & y_lim & z_lim]

quick_viz(positive_data)
plt.suptitle('Keep only above table')

total_drop = 600
per_axis_drop = total_drop//3
keep_rows = positive_data.z > 0 # Need a series of all True values
for axis in ['x', 'y', 'z']:
    top_axis = positive_data.nlargest(n=per_axis_drop//2, columns=axis).iloc[-1]
    bot_axis = positive_data.nsmallest(n=per_axis_drop//2, columns=axis).iloc[-1]
    keep_axis = (positive_data[axis]>bot_axis[axis]) & (positive_data[axis]<top_axis[axis])
    keep_rows = keep_rows & keep_axis

positive_data = positive_data[keep_rows]

quick_viz(positive_data)
plt.suptitle('Drop 100 in each direction')
# plt.show()

ids = list(set(negative_data.id.values).intersection(set(positive_data.id.values)))
columns_to_keep = ['id','x','y','z','rx','ry','rz']
positive_data = positive_data[positive_data.id.isin(ids)][columns_to_keep]
negative_data = negative_data[negative_data.id.isin(ids)]

# TODO: Preprocess the rotation too
positive_data.to_csv('matched_up_data/2percent_removed/matched_pos_data_in_table_coordinates.csv')
negative_data.to_csv('matched_up_data/2percent_removed/matched_neg_data_in_table_coordinates.csv')
