# -*- coding: utf-8 -*-
# !usr/bin/env python

# A script to drive calculation for ts search using VIBRON.

import re
import sys

import numpy as np


# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-GAUSSIAN-INTERFACE-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

# ACES fcm output files had neatly organized data, so using line numbers was more relevant.
# Here, things are scattered and messy and all over the place, so using re search is better.

def read_log_gaussian(filename):
    """Read Gaussian log file to extract single-point-energy and zero-point-energy

    Parameters
    ----------
    filename: str
        Gaussian log file name

    Returns
    -------
    float
        Single-point energy
    float
        Zero-point energy

    Raises
    ------
    Exception
        If log file is incomplete or erroneous and does not have energies

    """

    E0 = None
    ZPE = None

    try:
        f = open(filename, 'r')
    except EnvironmentError:
        print('Something not right with Gaussian log file.')
        print('IOError / OSError / WindowsError. Check. Fix. Rerun.')

    line = f.readline()
    while line != '':

        if 'SCF Done:' in line:
            E0 = float(line.split()[4])

        # Do NOT read the ZPE from the "E(ZPE)=" line, as this is the scaled version!
        # We will read in the unscaled ZPE and later multiply the scaling factor
        # from the input file

        elif 'Zero-point correction=' in line:
            ZPE = float(line.split()[2])
        elif '\\ZeroPoint=' in line:
            line = line.strip() + f.readline().strip()
            start = line.find('\\ZeroPoint=') + 11
            end = line.find('\\', start)
            ZPE = float(line[start:end])
        else:
            pass

        # Read the next line in the file
        line = f.readline()

    # Close file when finished
    f.close()

    if E0 is not None:
        if ZPE is not None:
            return E0, ZPE
        else:
            return E0
    else:
        raise Exception('Unable to find energy or zpe in Gaussian log file.')


def read_fchk_gaussian(filename, gaussian_version):
    """Read formatted checkpoint file from gaussian output

    Parameters
    ----------
    filename: str
        Name of the Gaussian checkpoint file
    gaussian_version: str
        Version of Gaussian (g09 or g16)

    Returns
    -------
    int
        Number of atoms
    array_like
        Array of atomic numbers
    array_like
        Array of atomic masses
    array_like
        Cartesian coordinates
    array_like
        Cartesian gradients
    array_like
        Cartesian force constant matrix (FCM)

    Raises
    ------
    Exception
        If the checkpoint file has an error!

    """
    if gaussian_version == 'g16':
        stuff = re.search(
            r'Atomic numbers\s+I\s+N=\s+(?P<num_atoms>\d+)'
            r'\n\s+(?P<anums>.*?)'
            'Nuclear charges.*?Current cartesian coordinates.*?\n(?P<coords>.*?)'
            'Number of symbols in /Mol/'
            '.*?Real atomic weights.*?\n(?P<masses>.*?)'
            'Atom fragment info.*?Cartesian Gradient.*?\n(?P<evals>.*?)'
            'Cartesian Force Constants.*?\n(?P<hess>.*?)'
            'Nonadiabatic coupling',
            open(filename, 'r').read(), flags=re.DOTALL)
    elif gaussian_version == 'g09':
        stuff = re.search(
            r'Atomic numbers\s+I\s+N=\s+(?P<num_atoms>\d+)'
            r'\n\s+(?P<anums>.*?)'
            'Nuclear charges.*?Current cartesian coordinates.*?\n(?P<coords>.*?)'
            'Force Field'
            '.*?Real atomic weights.*?\n(?P<masses>.*?)'
            'Atom fragment info.*?Cartesian Gradient.*?\n(?P<evals>.*?)'
            'Cartesian Force Constants.*?\n(?P<hess>.*?)'
            'Dipole Moment',
            open(filename, 'r').read(), flags=re.DOTALL)
    else:
        print('Gaussian version not defined. Provide g16 or g09. Will quit')
        sys.exit()

    if stuff is not None:

        # Atomic Number, Masses, Cartesian Geometry
        anums = list(map(int, stuff.group('anums').split()))
        anums = np.array(anums)
        masses = list(map(float, stuff.group('masses').split()))
        masses = np.array(masses)
        coords = list(map(float, stuff.group('coords').split()))
        coords = [coords[i:i + 3] for i in range(0, len(coords), 3)]
        coords = np.array(coords)

        natom = len(anums)  # get no of atoms by simply taking length of anums/masses array!

        evals = np.array(list(map(float, stuff.group('evals').split())), dtype=float)

        # Force Constant Matrix
        low_tri = np.array(list(map(float, stuff.group('hess').split())), dtype=float)
        one_dim = 3 * natom
        # force_constant_matrix = np.empty([one_dim, one_dim], dtype=float)
        # NOTE: np.empty does funny stuff, and caused me grief. As always,
        # it seems np.zeros is more reliable. Not gonna do PhD on it!
        force_constant_matrix = np.zeros([one_dim, one_dim], dtype=float)
        force_constant_matrix[np.tril_indices_from(force_constant_matrix)] = low_tri
        force_constant_matrix += np.tril(force_constant_matrix, -1).T

        return natom, anums, masses, coords, evals, force_constant_matrix

    else:
        raise Exception('No match found! Likely you provided a wrong fchk filename/path. Check. Try Again!')

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-END-GAUSSIAN-INTERFACE-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

