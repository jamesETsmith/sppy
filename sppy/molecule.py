#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
la = np.linalg
from sppy import atomic_data

class Molecule:
    '''
    Attributes:
        xyz (np.array): 2D array of coordinates (default in angstroms).
        atom (np.array): 1D array of atomic numbers.
        mass (np.array): 1D array of masses.
    '''

    def __init__(self, xyz, atom, mass=np.array([])):
        # TODO add conversion of lists to ndarrays

        self._xyz = xyz

        # If symbols given, convert to atomic number
        if isinstance(atom[0], int) != True:
            for i in range(len(atom)):
                for a in atomic_data.data:
                    if a[1] == atom[i]:
                        atom[i] = a[0]

        # Convert to ndarray
        if type(atom) != np.ndarray:
            atom = np.array(atom)
        self._atom = atom

        # Set masses as isotopically weighted avg if none given
        if mass.size == 0:
            self._mass = np.zeros((self._atom.size,))
            for i in range(self._mass.size):
                self._mass[i] = atomic_data.data[self._atom[i]][3]
        else:
            self._mass = mass


################################################################################

    def plot(self, ax):
        '''Plot the molecule on the given axis'''

        # Set the color array for the atoms
        colors = []
        size = []
        for an in self._atom:
            colors.append(atomic_data.data[an][4])
            size.append(atomic_data.data[an][5]**2/10)

        return ax.scatter(self._xyz[:,0], self._xyz[:,1], self._xyz[:,2],
            c=colors,s=size, depthshade=False)


################################################################################

def er_rotation(v1, v2):
    '''
    Euler-Rodrigues rotation of vector 1 to align with vector 2.
    Arguments:
        v1: vector that will be rotated
        v2: vector that we will rotate to (i.e. we will make v1 || to v2)
    Returns:
        r: 3x3 rotation matrix
    '''

    # Vector we will rotate about
    k = np.cross(v1,v2)
    k /= la.norm(k)

    # Angle we need to rotate
    th = np.arccos( np.dot(v1,v2)/(la.norm(v1)*la.norm(v2)) )

    # Euler/Rodrigues params
    # See https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    a = np.cos(th/2.)
    b = k[0] * np.sin(th/2.)
    c = k[1] * np.sin(th/2.)
    d = k[2] * np.sin(th/2.)

    print("CHECK %f = 1" % (a**2 + b**2 + c**2 + d**2))

    r = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

    return r


################################################################################

def rotate(k,th):
    '''
    Rotation around an axis.
    Arguments:
        k: axis to rotate around
        th: angle of rotation in radians
    Returns:
        r: 3x3 rotation matrix
    '''
    # Make sure k is a unit vector
    k /= la.norm(k)

    # Euler/Rodrigues params
    # See https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
    a = np.cos(th/2.)
    b = k[0] * np.sin(th/2.)
    c = k[1] * np.sin(th/2.)
    d = k[2] * np.sin(th/2.)
    r = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

    return r

################################################################################

def rotate_dihedral(p1, p2, th, npts, rotor, xyz):
    '''
    Rotate dihedral angle in molecule. Dihedral bond is always aligned to
    z-axis.

    Arguments:
        p1 (int): Index of pivot atom 1 in xyz array.
        p2 (int): Index of pivot atom 2 in xyz array.
        th (float): Angle to rotate rotor by.
        npts (int): Number of rotations to perform.
        rotor (ndarray): List of indices of atoms that will be rotated.
        xyz (ndarray): 2D ndarray of atom coordinates.

    Returns:

    '''

    # Translate pivot 1 to the origin.
    xyz -= xyz[p1]

    # Rotate to align dihedral bond along z-axis.
    v1 = xyz[p2]
    v2 = np.array([0,0,1.])
    r_mat = er_rotation(v1,v2)
    xyz = np.einsum('ij,kj->ik',xyz,r_mat)

    #
    rotated_xyz = np.zeros((npts+1,xyz.shape[0],xyz.shape[1]))
    rotated_xyz[0,:,:] = xyz

    for i in range(1,npts+1):
        rotated_xyz[i,:,:] = rotated_xyz[i-1,:,:]
        rtr = rotated_xyz[i,rotor,:]
        r = rotate(v2, th)
        rtr = np.einsum('ij,kj->ik',rtr,r)
        rotated_xyz[i,rotor,:] = rtr

    return rotated_xyz
################################################################################



if __name__ == '__main__':
    atoms = ['C','H','H','H','C','H','C','H','C','H','H','H']
    xyz = np.array([[-4.78885668,    1.36034493,    0.00000000],
                 [-4.43220225,    0.35153493,    0.00000000],
                 [-4.43218384,    1.86474312,   -0.87365150],
                 [-5.85885668,    1.36035811,    0.00000000],
                 [-4.27551446,    2.08630120,    1.25740497],
                 [-3.66047964,    1.56454028,    1.96053917],
                 [-4.60006716,    3.38370673,    1.47642888],
                 [-4.24040385,    3.88279822,    2.35190018],
                 [-5.48474662,    4.14217805,    0.46962030],
                 [-5.63159701,    5.14610950,    0.80940938],
                 [-5.00532224,    4.15589711,   -0.48686496],
                 [-6.43200027,    3.65151711,    0.38678096] ])

    # mol = Molecule(xyz,atoms)

    npts = 24
    rotor = np.arange(6)
    rotated_xyz = rotate_dihedral(4,6,2*np.pi/npts,npts,rotor,xyz)

    # ffmpeg -framerate 24 -i %d.png test.mpeg
    i = 0
    for rxyz in rotated_xyz:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20., azim=45)
        mol = Molecule(rxyz,atoms)
        mol.plot(ax)
        plt.savefig("%d" % i)
        i += 1
