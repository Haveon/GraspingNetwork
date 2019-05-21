import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

calculate = False

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

data = pd.read_csv('eulers.csv')

if calculate:
    acc = np.zeros_like(THETA).flatten()
    for i in range(data.shape[0]):
        point = data.iloc[i][['theta', 'phi', 'psi']].values
        dist = distance(point)
        index = np.argmin(dist.flatten())
        acc[index]+=1

    tmp = pd.Series(acc)
    tmp.to_csv('ACC.csv', index=False)

tmp = pd.Series(np.loadtxt('ACC.csv', delimiter=',').flatten())
print('numVox\tnumPoint\tpercentCov')
for i in [1000,1500,2000,2500,3000,3504]:
    numPoint = tmp.nlargest(i).sum()
    perCov = numPoint/tmp.sum()
    print(f'{i}\t{numPoint}\t\t{perCov:.4}')

acc = tmp.values
mask = acc>3
print(mask.sum())
print(acc[mask].sum()/acc.sum())
coords = np.stack([THETA.flatten(), PHI.flatten(), PSI.flatten()], axis=-1)
voxels = coords[mask]
np.savetxt('rot_vox.tsv', voxels)

th, ph, ps = voxels[:,0], voxels[:,1], voxels[:,2]
th = (th+2*np.pi)%(2*np.pi) - np.pi
ps = (ps+2*np.pi)%(2*np.pi) - np.pi

ax = plt.figure().add_subplot(111, projection='3d'); ax.set_aspect(1)
ax.scatter(th, ph, ps)
plt.show()
