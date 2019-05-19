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

# ----------------------------------------------------------------------------
import pandas as pd
if False:
    viz()
    plt.show()

positive_data = pd.read_csv('matched_up_data/2percent_removed/matched_pos_data_in_camera_coordinates.csv')
negative_data = pd.read_csv('matched_up_data/2percent_removed/matched_neg_data_in_camera_coordinates.csv')

rotation = positive_data[['id','rx', 'ry', 'rz', 'rot_mag']].values
ids,rx,ry,rz,mag = rotation[:,0],rotation[:,1],rotation[:,2],rotation[:,3],rotation[:,4]
eulers = np.zeros([rx.shape[0],3])
for i in range(rx.shape[0]):
    axis = np.array([rx[i],ry[i],rz[i]])
    angle = mag[i]
    if angle > np.pi:
        angle = 2*np.pi - angle
        axis = -1*axis
    q = make_quaternion_from_axis_angle(axis, angle)
    eul = make_euler_angles_from_quaternion(q)
    eulers[i] = eul

xlim = [-2.5,2.2]
ylim = [1.30, 1.60]

def rotate_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]])
def rotate_y(theta):
    return np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
def rotate_x(theta):
    return np.array([[1,0,0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])

eulers = pd.DataFrame({'theta': eulers[:,0], 'phi': eulers[:,1], 'psi': eulers[:,2], 'x': positive_data.x.values, 'y': positive_data.y.values, 'z': positive_data.z.values})
mask = (eulers.theta>xlim[0])&(eulers.theta<xlim[1])&(eulers.phi>ylim[0])&(eulers.phi<ylim[1])
red_eulers = eulers[mask]
print(red_eulers.shape)
test_angles = red_eulers.sample(1000).values

rot1 = rotate_x(np.deg2rad(-102))
rot2 = rotate_y(np.deg2rad(-0.9))
rot3 = rotate_z(np.deg2rad(-135))
R = rot3 @ rot2 @ rot1

offset = np.array([-0.353906, 0.286732, 0.2275])
# ax = plt.figure().add_subplot(111, projection='3d')
tmp_r = np.zeros([test_angles.shape[0],3,3])
count=0
for theta,phi,psi,x,y,z in test_angles:
    rot = R @ rotate_z(psi) @ rotate_x(phi) @ rotate_z(theta)
    o = (rot @ np.array([[1,0,0],[0,1,0],[0,0,1]])).T
    # p = (R @ np.array([[x],[y],[z]])).T + offset
    # for i in range(3):
    #     vec = 0.005*rot[:,i] + p
    #     ax.plot(*np.concatenate([p,vec]).T,c=('black' if i==2 else ('blue' if i==1 else 'red')))
    tmp_r[count] = rot
    count+=1

# for x,y,z in zip([0.05,0,0],[0,0.05,0],[0,0,0.05]):
#     ax.plot([0,x],[0,y],[0,z])
# plt.title('pos')
# ax.set_aspect(1)
# ax.set_xlim([-0.1,0.1])
# ax.set_ylim([-0.1,0.1])
# ax.set_zlim([0,0.2])
# plt.show()

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(*tmp_r[:,:,0].T)
plt.title('X')

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(*tmp_r[:,:,1].T)
plt.title('Y')

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(*tmp_r[:,:,2].T)
plt.title('Z')
plt.show()

# theta_anchor, phi_anchor = fib_spiral(1024)
# gamma_anchor = np.linspace(0,2*np.pi,130)[1:-1]
# positive_orient = assign_sphere_position(positive_data, theta_anchor, phi_anchor, gamma_anchor)
# negative_orient = assign_sphere_position(negative_data, theta_anchor, phi_anchor, gamma_anchor)
#
# np.savetxt('sphere/2percent_removed/positive_sphere_compact.csv',
#             positive_orient, fmt='%d', delimiter=',', header='id,sphere,gamma')
# np.savetxt('sphere/2percent_removed/negative_sphere_compact.csv',
#             negative_orient, fmt='%d', delimiter=',', header='id,sphere,gamma')
# np.savetxt('sphere/2percent_removed/sphere_centers.csv', np.stack([theta_anchor,phi_anchor]).T,
#             delimiter=',', header='theta,phi')
# np.savetxt('sphere/2percent_removed/gamma.csv', gamma_anchor,
#             delimiter=',', header='anchor')
