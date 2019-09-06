import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GOLDEN = (1+5**0.5) / 2

def frac_part(x):
    return np.modf(x)[0]

def lambert_equal_area_proj(xs,ys):
    return 2*np.pi*ys, np.arccos(2*xs-1)

def fib_spiral(N):
    i = np.arange(1,N-1)
    xs,ys = np.zeros(N), np.zeros(N)
    xs[1:N-1], ys[1:N-1] = (i+6)/(N+11), frac_part(i/GOLDEN)
    xs[N-1] = 1

    theta, phi = lambert_equal_area_proj(xs,ys)
    return theta,phi

def spherical_to_cartesian(r,theta,phi):
    """
    Assumes math convention: theta is the azimuthal angle, phi is the polar angle
    """
    return r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)

def cartesian_to_spherical(x,y,z):
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/r)
    return r, theta, phi

def make_euler_angles_from_quaternion(q):
    q0,qx,qy,qz = q
    theta = np.arctan2( qx*qz + qy*q0, -(qy*qz - qx*q0) )
    phi   = np.arccos(-qx*qx - qy*qy + qz*qz + q0*q0)
    psi   = np.arctan2(qx*qz - qy*q0, qy*qz + qx*q0)
    return np.array([theta, phi, psi])

def make_quaternion_from_euler_angles(eul):
    theta, phi, psi = eul
    th,ph,ps = theta/2, phi/2, psi/2

    qx = np.cos(th-ps)*np.sin(ph)
    qy = np.sin(th-ps)*np.sin(ph)
    qz = np.sin(th+ps)*np.cos(ph)
    q0 = np.cos(th+ps)*np.cos(ph)
    return np.array([q0,qx,qy,qz])

def make_quaternion_from_axis_angle(axis, angle):
    """
    Angle must be in radians
    Axis must have norm 1
    """
    if angle > np.pi:
        angle = 2*np.pi - angle
        axis = -1*axis
    q0 = np.cos(angle/2)
    q = axis*np.sin(angle/2)
    return np.array([q0,q[0],q[1],q[2]])

def viz():
    N = 1024
    theta,phi = fib_spiral(N)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs,ys,zs = spherical_to_cartesian(np.ones(N), theta, phi)
    ax.scatter(xs,ys,zs)
    ax.set_aspect(1)

def assign_sphere_position(data, theta_anchor, phi_anchor, gamma_anchor):
    rotation = data[['id','rx', 'ry', 'rz', 'rot_mag']].values
    ids,rx,ry,rz,mag = rotation[:,0],rotation[:,1],rotation[:,2],rotation[:,3],rotation[:,4]
    indeces = np.zeros_like(rx)
    gindeces = np.zeros_like(rx)
    for i in range(rx.shape[0]):
        r, theta, phi = cartesian_to_spherical(rx[i], ry[i], rz[i])
        index = np.argmin(np.sqrt((theta_anchor - theta)**2 + (phi_anchor - phi)**2), axis=-1)
        indeces[i] = index
        gindex = np.argmin( np.sqrt((mag[i] - gamma_anchor)**2), axis=-1 )
        gindeces[i] = gindex

    orient = np.stack([ids,indeces,gindeces]).T
    return orient

d_ang = np.pi/32
nodes_th = np.arange(-np.pi, np.pi, d_ang)
nodes_ph = np.arange(     0, np.pi, d_ang)
nodes_ps = np.arange(-np.pi, np.pi, d_ang)

THETA, PHI, PSI = np.meshgrid(nodes_th, nodes_ph, nodes_ps)

def distance(point):
    d_th = np.abs(THETA - point[0])
    d_ph = np.abs(PHI - point[1])
    d_ps = np.abs(PSI - point[2])
    d_th = np.where(d_th>np.pi, 2*np.pi - d_th, d_th)
    d_ph = np.where(d_ph>(np.pi/2), np.pi - d_ph, d_ph)
    d_ps = np.where(d_ps>np.pi, 2*np.pi - d_ps, d_ps)
    ss = d_th**2 + d_ph**2 + d_ps**2
    return np.sqrt(ss)

flat_theta = THETA.flatten()
flat_phi   = PHI.flatten()
flat_psi   = PSI.flatten()
flat_coords = np.stack([flat_theta, flat_phi, flat_psi], axis=-1)

# ----------------------------------------------------------------------------
from collections import defaultdict
import pandas as pd
if False:
    viz()
    plt.show()

positive_data = pd.read_csv('matched_up_data/2percent_removed/matched_pos_data_in_camera_coordinates.csv')
negative_data = pd.read_csv('matched_up_data/2percent_removed/matched_neg_data_in_camera_coordinates.csv')

rotation = positive_data[['id','rx', 'ry', 'rz', 'rot_mag']].values
ids,rx,ry,rz,mag = rotation[:,0],rotation[:,1],rotation[:,2],rotation[:,3],rotation[:,4]
eulers = np.zeros([rx.shape[0],3])
keep_dict = defaultdict(list)
for i in range(rx.shape[0]):
    axis = np.array([rx[i],ry[i],rz[i]])
    angle = mag[i]
    if angle > np.pi:
        angle = 2*np.pi - angle
        axis = -1*axis
    q = make_quaternion_from_axis_angle(axis, angle)
    eul = make_euler_angles_from_quaternion(q)
    eulers[i] = eul
    dist = distance(eul)
    index = np.argmin(dist.flatten())
    keep_dict[tuple(flat_coords[index])].append(ids[i])

data = pd.DataFrame({'id': ids,
                     'x': positive_data.x, 'y': positive_data.y, 'z': positive_data.z,
                     'theta': eulers[:,0], 'phi': eulers[:,1], 'psi': eulers[:,2]})

ids_to_keep = []
keys_to_keep = []
voxel_index = []
i = 0
for key in keep_dict:
    pop = len(keep_dict[key])
    if pop>3:
        ids_to_keep += keep_dict[key]
        keys_to_keep.append(key)
        voxel_index+=[i]*pop
        i+=1

rot_vox_centers = np.array(keys_to_keep)
np.savetxt('sphere/2percent_removed/rot_voxel_center.csv', rot_vox_centers, delimiter=',', header='theta,phi,psi')
id_mask = data.id.isin(ids_to_keep)
rot_data = data[id_mask]

position_cubes = pd.read_csv('cubes/2percent_removed/positive_position_cube_compact.csv')
cubes = position_cubes[id_mask][['x', 'y', 'z']].values
binned_data = np.stack([ids_to_keep, voxel_index, cubes[:,0], cubes[:,1], cubes[:,2]], axis=-1)
np.savetxt('sphere/2percent_removed/rotation_targets.csv', binned_data, delimiter=',', header='ids,rot_voxel,cube_x,cube_y,cube_z', fmt='%d')
