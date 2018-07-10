#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from sppy import molecule
import os
la = np.linalg

atoms = ['C','H','H','C','H','C','H','C','H','C','H','N','H','H']
xyz = np.array([[-2.92664500,   -0.29472900,   -0.00000100],
                [-3.58316600,   -1.15638900,   -0.00000200],
                [-3.39812100,    0.68248500,   -0.00000100],
                [-1.58632400,   -0.43050700,    0.00000100],
                [-1.15709900,   -1.42666900,    0.00000100],
                [-0.73522400,    0.72141200,    0.00000200],
                [-1.25010900,    1.67892200,    0.00000400],
                [0.63831800,    0.77919800,   -0.00000100],
                [1.11352300,    1.75464200,   -0.00000300],
                [1.46919200,   -0.35527300,    0.00000000],
                [1.04484400,   -1.35363500,    0.00000100],
                [2.77932900,   -0.30449400,    0.00000000],
                [3.34042200,   -1.14635300,    0.00000100],
                [3.27850100,    0.57784900,   -0.00000100]])

npts = 18
rotor = np.arange(7)
rotated_xyz = molecule.rotate_dihedral(5,7,2*np.pi/npts,npts,rotor,xyz)

# Plot each rotated molecule
if not os.path.isdir("figures"):
    os.mkdir("figures")

n_repeats = 10
i = 0
plt.figure()
for rxyz in rotated_xyz:
    # plt.scatter(rxyz[:,0],rxyz[:,1])
    # print(rxyz)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    mol = molecule.Molecule(rxyz,atoms)
    mol.plot(ax)
    
    # Set the perspective and axis limits to make a smoothe movie
    ax.view_init(elev=20., azim=45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim3d(-4,4)
    ax.set_ylim3d(-4,4)
    ax.set_zlim3d(-4,4)

    # Adjusting framerate with number of images used
    for j in range(n_repeats):
        plt.savefig("figures/%d" % i)
        i += 1

# Combine w/
# $ ffmpeg -framerate 24 -i %d.png test.mpeg
