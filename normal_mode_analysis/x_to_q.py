# -*- coding: utf-8 -*-
# !usr/bin/env python

import numpy as np

from normal_mode_analysis.physical_constants import fred

# Print Precision!
np.set_printoptions(precision=8, suppress=True)


def get_x_to_q_transformation_matrix(natom, amass, nmode, ref_cart, ref_freq, l_ref_mat, is_dimensionless):
    """Transformation matrices for X -> Q and Q -> X transformation

    Parameters
    ----------
    natom: int
        Number of atoms
    amass: array_like
        Array of atomic masses
    nmode: int
        Number of normal modes
    ref_cart: array_like
        Reference cartesian coordinates
    ref_freq: array_like
        Reference vibrational frequencies
    l_ref_mat: array_like
        Reference normal modes
    is_dimensionless: bool
        Dimensionless coordinates or not

    Returns
    -------
    array_like
        Cratesian to normal coordinate transformation matrix
    array_like
        Normal coordinate to cartesian transformation matrix

    Notes
    -----
    Q -> X: M^(-0.5) * l_ref_mat ("D_mat")
    X -> Q = (D.T * D)^-1 * D.T  ("D_dagger_mat")

    """

    # Get the matrix of atomic masses 
    mass_matrix_sqrt_div = np.diag(np.repeat(1.0 / np.sqrt(amass), 3))

    # -------------
    # MASS-WEIGHTED
    # -------------

    # 1. Q -> X
    D_mat = np.dot(mass_matrix_sqrt_div, l_ref_mat)

    # 2. X -> Q
    DTDI = np.linalg.inv(np.dot(D_mat.T, D_mat))
    D_dagger_mat = np.dot(DTDI, D_mat.T)

    # -------------
    # DIMENSIONLESS
    # -------------

    # 3. The Transformation Matrix : D~ [MCTDH Notation] (3N - 3, 3N)
    Qdmfs = np.zeros(np.shape(l_ref_mat)).T
    for alpha in range(nmode):
        for i in range(natom):
            for j in range(3):
                k = 3 * i + j
                Qdmfs[alpha][k] = (fred * np.sqrt(ref_freq[alpha]) * np.sqrt(amass[i])) * l_ref_mat.T[alpha][k]

    # 4. The Transformation Matrix: D~' [MCTDH Notation] (3N, 3N - 3)
    Xdmfs = np.zeros(np.shape(l_ref_mat))
    for alpha in range(nmode):
        for i in range(natom):
            for j in range(3):
                k = 3 * i + j
                Xdmfs[k][alpha] = l_ref_mat[k][alpha] / (fred * np.sqrt(ref_freq[alpha]) * np.sqrt(amass[i]))

    # ====================================
    # Write to file along with a test case
    # ====================================

    # Open file
    gwrite = open('transform_cartesian_normal', 'w')

    gwrite.write(str(natom) + '\n')
    for i in range(natom):
        gwrite.write("{:20.8f}".format(amass[i]))
    gwrite.write('\n')
    gwrite.write(str(nmode) + '\n')

    # Write reference geometry in cartesian coordinates
    gwrite.write('Cartesian Reference Geometry\n')
    for i in range(natom):
        for j in range(3):
            gwrite.write("{:20.8f}".format(ref_cart[i][j]))
        gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write transformation matrix for Q --> X transformation
    gwrite.write('Transformation Matrix for Q to X\n')
    for i in range(3 * natom):
        for j in range(nmode):
            if not is_dimensionless:
                gwrite.write("{:20.8f}".format(D_mat[i][j]))
            else:
                gwrite.write("{:20.8f}".format(Xdmfs[i][j]))

        gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write transformation matrix for X --> Q transformation
    gwrite.write('Transformation Matrix for X to Q\n')
    for i in range(nmode):
        for j in range(3 * natom):
            if not is_dimensionless:
                gwrite.write("{:20.8f}".format(D_dagger_mat[i][j]))
            else:
                gwrite.write("{:20.8f}".format(Qdmfs[i][j]))
        gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write Reference Normal Modes
    gwrite.write('Reference Normal Modes\n')
    for i in range(3 * natom):
        for j in range(nmode):
            gwrite.write("{:20.8f}".format(l_ref_mat[i][j]))
        gwrite.write('\n')
    gwrite.write('\n')
    gwrite.write('\n')

    # Write reference frequencies
    gwrite.write('Reference Frequencies\n')
    for i in range(nmode):
        gwrite.write("{:20.8f}".format(ref_freq[i]))
    gwrite.write('\n')
    gwrite.write('\n')

    gwrite.close()

    #  Finally Return
    if not is_dimensionless:
        return D_mat, D_dagger_mat
    else:
        return Xdmfs, Qdmfs

