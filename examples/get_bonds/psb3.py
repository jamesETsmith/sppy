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

mol = molecule.Molecule(xyz,atoms)
mol.get_bonds_by_distance()

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
mol.plot(ax)
ax.set_xlim3d(-4,4)
ax.set_ylim3d(-4,4)
ax.set_zlim3d(-4,4)
plt.show()
