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
    return r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)

if __name__ == '__main__':
    N = 1024
    theta,phi = fib_spiral(N)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs,ys,zs = spherical_to_cartesian(np.ones(N), theta, phi)
    ax.scatter(xs,ys,zs)
    ax.set_aspect(1)
    plt.show()
