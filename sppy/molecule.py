#!/usr/bin/env python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
la = np.linalg
from sppy import atomic_data

class Molecule:
    '''
    Attributes:
        xyz (ndarray): 2D array of coordinates (default in angstroms).
        atom (ndarray): 1D array of atomic numbers.
        mass (ndarray): 1D array of masses.
        bonds (ndarray): 2D array of bonds.
    '''

    def __init__(self, xyz, atom, mass=np.array([]), bonds=np.array([])):
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

        # Bonds
        self._bonds = bonds

################################################################################

    def plot(self, ax, show_legend=False):
        '''Plot the molecule on the given axis'''

        # Set the color array for the atoms
        colors = []
        size = []
        symbs = []

        for an in self._atom:
            colors.append(atomic_data.data[an][4])
            size.append(atomic_data.data[an][5]*10)
            symbs.append(atomic_data.data[an][1])

        for i in range(self._bonds.shape[0]):
            xs = np.zeros((2,)); ys = np.zeros((2,)); zs = np.zeros((2,))

            xs[0] = self._xyz[self._bonds[i,0]][0]
            xs[1] = self._xyz[self._bonds[i,1]][0]

            ys[0] = self._xyz[self._bonds[i,0]][1]
            ys[1] = self._xyz[self._bonds[i,1]][1]

            zs[0] = self._xyz[self._bonds[i,0]][2]
            zs[1] = self._xyz[self._bonds[i,1]][2]

            ax.plot(xs, ys, zs, c='darkgray', linewidth=5, zorder=1)

        # ax.scatter(self._xyz[:,0], self._xyz[:,1], self._xyz[:,2],
        #     c=colors,s=size, depthshade=False, zorder=10, label=symbs)

        # Plot the atoms one at a time to label them properly
        for i in range(self._xyz.shape[0]):
            # If the element has already been labelled, don't label it again
            if symbs[i] in symbs[:i]:
                ax.scatter(self._xyz[i,0], self._xyz[i,1], self._xyz[i,2],
                    c=colors[i],s=size[i], depthshade=False, zorder=10)
            else:
                ax.scatter(self._xyz[i,0], self._xyz[i,1], self._xyz[i,2],
                    c=colors[i],s=size[i], depthshade=False, zorder=10,
                    label=symbs[i])

        if show_legend:
            legend = ax.legend(labelspacing=2) # Prevent overlapping symbols
            legend.get_frame().set_facecolor('#00FFCC')
            legend.get_frame().set_alpha(1)


################################################################################

    def get_bonds_by_distance(self, cutoff=1.6):
        '''
        Determine the bonded atoms by a distance provied. Cutoff is inclusive.
        '''
        bonds = []
        for i in range(self._xyz.shape[0]):
            for j in range(i+1,self._xyz.shape[0]):
                r = la.norm(self._xyz[i]-self._xyz[j])
                if r <= cutoff:
                    bonds.append([i,j])

        self._bonds = np.array(bonds)

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

def parse_pyscf_atom(atom):
    '''
    Parses the mol.atom attribute of PySCF molecule object to get the
    coordinates and atoms and returns a Molecule object.

    .. note::

        Currently can only parse when mol.atom has the following form:

        mol.atom = """
        O 0. 0. 0.
        H 0. 1. 0.
        H 0. 0. 1.
        """

    atom (string): atom attribute of PySCF mol object
    '''

    atom_split = atom.split()
    atoms = []
    xyz = np.zeros(( int(len(atom_split)/4), 3 ))

    for i in range(xyz.shape[0]):
        atoms.append(atom_split[4*i])
        xyz[i,0] = float(atom_split[4*i+1])
        xyz[i,1] = float(atom_split[4*i+2])
        xyz[i,2] = float(atom_split[4*i+3])

    return Molecule(xyz,atoms)

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
    n_repeats = 10
    i = 0
    for rxyz in rotated_xyz:
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20., azim=45)
        mol = Molecule(rxyz,atoms)
        mol.plot(ax)

        # Slowing down the framerate in ffmpeg can be a pain
        for j in range(n_repeats):
            plt.savefig("figures/%d" % i)
            i += 1
