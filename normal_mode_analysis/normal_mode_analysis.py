# -*- coding: utf-8 -*-
# !usr/bin/env python

# @Author: Prateek Goel
# @Date: 03/05/2017
# @Email: prateik.goel@gmail.com
# @Last modified by: Prateek

import numpy as np

from normal_mode_analysis.gaussian_interface import read_log_gaussian, read_fchk_gaussian
from normal_mode_analysis.input_nma import project_translation, project_rotation, hfreq_cm
from normal_mode_analysis.physical_constants import au_to_ev

# Print Precision!
np.set_printoptions(precision=8, suppress=True)


# sys.tracebacklimit=0

# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-
# x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-CLASS-SECTION-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-

class Molecule(object):
    """
    Abstraction of a molecule, primarily based on the cartesian geometry and hessian.
    Associated methods will perform the harmonic vibrational analysis and the rigid
    rotor rotational analysis. Appropriate coordinate transformations will be perfomed.
    
    Attributes
    ----------
    natom: int
        Number of atoms
    amass: array_like
        Array of atomic masses
    eq_geom_cart: array_like
        Molecular geometry in cartesian coordinates
    force_constant_matrix: array_like
        Force Constant Matrix in cartesian coordinates
    energy: float
        Single point electronic energy at the given geometry
    zpe: float
        The zero point energy as reported by Gaussian / ACES
    gradient: array_like
        Gradient vector at the corresponding geometry
    hessian: array_like
        Hessian matrix at the corresponding geometry
    frequencies: array_like
        Harmonic Vibrational Frequencies (in Wavenumbers)
    mwcart: array_like
        Mass-weighted cartesian goemetry
    com: array_like
        Centre-of-Mass coordinates
    moi: array_like
        Moment-of-Inertia Tensor
    PointGroup: str
        Point Group Symmetry
    RotationalConstants: array_like
        [A, B, C] associated with Ia, Ib, Ic
    Lcart: array_like
        Cartesian Normal Coordinates
    Lmwc: array_like
        Mass-weighted Cartesian Normal Coordinates
    Ldmfs: array_like
        Dimensionless frequency scaled Normal Coordinates
    Ltransrot: array_like
        Constructed Translation and/or Rotation Eigenvectors
    EckartMatrix: array_like
        Eckart Rotation Matrix for a given reference geometry
    Duschinsky: array_like
        Duschinsky rotation matrix for a given reference geometry
    Displacement: array_like
        Displacement vector for a given reference geometry

    """

    def __init__(self, natom, amass, eq_geom_cart, force_constant_matrix):

        self.natom = natom
        self.amass = amass
        self.eq_geom_cart = eq_geom_cart
        self.force_constant_matrix = force_constant_matrix

        self.mwcart = None
        self.com = None
        self.moi = None
        self.energy = None
        self.zpe = None
        self.hessian = None
        self.frequencies = None

        self.PointGroup = None
        self.RotationalConstants = None

        self.Lcart = None
        self.Lmwc = None
        self.Ldmfs = None
        self.Ltransrot = None

        self.EckartMatrix = None
        self.Duschinsky = None
        self.Displacement = None

        # TODO: Linear molecules
        if project_translation and project_rotation:
            self.nmode = 3 * self.natom - 6
        elif project_translation and not project_rotation:
            self.nmode = 3 * self.natom - 3
        elif not project_translation and project_rotation:
            self.nmode = 3 * self.natom - 3
        else:
            self.nmode = 3 * self.natom

        self.gradient = np.zeros(self.nmode)

    def get_mass_weighted_cartesian_coordinates(self):
        """Function to compute mass-weighted Cartesian coordinates

        Returns
        -------
        array_like
            mass-weighted Cartesian coordinates

        """

        mwcart = np.zeros((self.natom, 3))

        for i in range(self.natom):
            mwcart[i][0] = np.sqrt(self.amass[i]) * self.eq_geom_cart[i][0]
            mwcart[i][1] = np.sqrt(self.amass[i]) * self.eq_geom_cart[i][1]
            mwcart[i][2] = np.sqrt(self.amass[i]) * self.eq_geom_cart[i][2]

        self.mwcart = mwcart

        return mwcart

    def get_centre_of_mass_mwc(self):
        """Function to compute centre of mass

        Returns
        -------
        array_like
            Centre of mass

        """

        TotMass = np.sum(self.amass)
        CentreMass = np.zeros(3)

        for i in range(self.natom):
            CentreMass[0] += np.sqrt(self.amass[i]) * self.mwcart[i][0]
            CentreMass[1] += np.sqrt(self.amass[i]) * self.mwcart[i][1]
            CentreMass[2] += np.sqrt(self.amass[i]) * self.mwcart[i][2]

        CentreMass = CentreMass / TotMass
        self.com = CentreMass

        return CentreMass

    def get_inertia_tensor(self):
        """Function to compute moment of inertia tensor

        Returns
        -------
        array_like
            moment of inertia tensor

        """

        Inertia_Tensor = np.zeros((3, 3))

        for i in range(self.natom):
            Inertia_Tensor[0][0] += self.mwcart[i][1] * self.mwcart[i][1] + self.mwcart[i][2] * self.mwcart[i][2]
            Inertia_Tensor[1][1] += self.mwcart[i][0] * self.mwcart[i][0] + self.mwcart[i][2] * self.mwcart[i][2]
            Inertia_Tensor[2][2] += self.mwcart[i][0] * self.mwcart[i][0] + self.mwcart[i][1] * self.mwcart[i][1]
            Inertia_Tensor[0][1] += self.mwcart[i][0] * self.mwcart[i][1]
            Inertia_Tensor[0][2] += self.mwcart[i][0] * self.mwcart[i][2]
            Inertia_Tensor[1][2] += self.mwcart[i][1] * self.mwcart[i][2]

        Inertia_Tensor[1][0] = Inertia_Tensor[0][1]
        Inertia_Tensor[2][0] = Inertia_Tensor[0][2]
        Inertia_Tensor[2][1] = Inertia_Tensor[1][2]

        self.moi = Inertia_Tensor

        return Inertia_Tensor

    @property
    def get_eckart_frame_self(self):
        """Function to put the molecule in its own Eckart frame

        Returns
        -------

        """

        # Check if CM at origin, and Inertia_Tensor Diagonal. If yes: exit
        # If no: translate CM to origin, diagonalize IT, and rotate coordinates

        tolerance = 1e-5

        # Always put COM at origin (even if it already is, does not hurt)
        # for i in range(self.natom):
        #    self.eq_geom_cart[i:,] = self.eq_geom_cart[i:,] - self.com

        self.eq_geom_cart = self.eq_geom_cart - self.com.T
        self.com = np.zeros(3)

        Ival = np.zeros(3)
        Ivec = np.zeros((3, 3))

        # Check for MOI
        for i in range(3):
            for j in range(3):
                if i != j:
                    if self.moi[i, j] > tolerance:
                        Ival, Ivec = np.linalg.eigh(self.moi)
                        break
                    else:
                        Ival, Ivec = np.linalg.eigh(self.moi)
                else:
                    pass

        self.moi = np.diag(Ival)

        # Rotate coordinates to Principal Axes
        self.eq_geom_cart = np.dot(self.eq_geom_cart, Ivec)

        return

    @property
    def harmonic_vibrational_analysis(self):
        """Harmonic Vibrational Analysis to get the normal modes

        Perform the normal mode analysis starting from the 3N x 3N Force Constant Matrix:

        - project out translations and/or rotations as requested
        - mass-weight force constant matrix, diagonalize (omega, L)
        - get frequencies in cm-1 (sqrt(omega)*5140.48)

        Returns
        -------
        array_like
            Vibrational frquencies in wavenumbers
        array_like
            Normal modes

        """
        # Get the matrix of atomic masses
        mass_matrix_sqrt_div = np.diag(np.repeat(1.0 / np.sqrt(self.amass), 3))

        # calculate the center of mass in cartesian coordinates
        xyzcom = self.eq_geom_cart - self.com.T

        # Initialize (3N, 6) array for Translation and Rotation
        Dmat = np.zeros((3 * self.natom, 6), dtype=float)

        #####################################################
        # Construct Eigenvectors correspoding to Translation#
        #####################################################

        for i in range(3):
            for k in range(self.natom):
                for alpha in range(3):
                    if alpha == i:
                        Dmat[3 * k + alpha, i] = np.sqrt(self.amass[k])
                    else:
                        pass

        ###################################################
        # Construct Eigenvectors correspoding to Rotation #
        ###################################################

        # 1. Get Inertia Tensor and Diagonalize
        Ival, Ivec = np.linalg.eigh(self.moi)

        # 2. Construct Pmat
        Pmat = np.dot(xyzcom, Ivec)

        # 3. Construct Rotational Normal Coordinates
        for i in range(self.natom):
            for j in range(3):
                Dmat[3 * i + j, 3] = (Pmat[i, 1] * Ivec[j, 2] - Pmat[i, 2] * Ivec[j, 1]) * np.sqrt(self.amass[i])
                Dmat[3 * i + j, 4] = (Pmat[i, 2] * Ivec[j, 0] - Pmat[i, 0] * Ivec[j, 2]) * np.sqrt(self.amass[i])
                Dmat[3 * i + j, 5] = (Pmat[i, 0] * Ivec[j, 1] - Pmat[i, 1] * Ivec[j, 0]) * np.sqrt(self.amass[i])

        ##################################################################################
        # Set the orthonormalized Translation-Rotation Eigenvectors to attribute Ltransrot
        ##################################################################################

        Translation = Dmat[:, 0:3]
        Rotation = Dmat[:, 3:6]

        # Separately orthonormalize translation and rotation
        Dtrans, xxx = np.linalg.qr(Translation)
        Drot, xxx = np.linalg.qr(Rotation)

        LTR = np.zeros((3 * self.natom, 6), dtype=float)
        LTR[:, 0:3] = Dtrans
        LTR[:, 3:6] = Drot

        self.Ltransrot = Drot

        # Mass-weight the force constant matrix
        mw_fcm = np.dot(mass_matrix_sqrt_div, np.dot(self.force_constant_matrix, mass_matrix_sqrt_div))

        # Project out Rotation and Translation from Hessian
        Imat = np.eye(LTR.shape[0])
        llt = np.dot(LTR, LTR.T)
        proj_trans_rot_hessian = np.dot(Imat - llt, np.dot(mw_fcm, Imat - llt))
        rphval, rphvec = np.linalg.eigh(proj_trans_rot_hessian)

        # SORT OUT ALL -VE FREQUENCIES
        all_index_0 = np.where(abs(rphval) < 1e-4)[0]
        eigvals_0 = rphval[all_index_0]
        eigvec_0 = rphvec[:, all_index_0]

        # A cleaner solution?
        rphval = np.delete(rphval, all_index_0, axis=0)
        rphvec = np.delete(rphvec, all_index_0, axis=1)
        rphval = np.concatenate([eigvals_0, rphval])
        rphvec = np.concatenate([eigvec_0, rphvec], axis=1)

        vib_freq_cm = np.sqrt(abs(rphval[6:])) * hfreq_cm
        Lmwc = rphvec[:, 6:3 * self.natom]

        # NORMAL MODES - SET ATTRIBUTE
        self.Lmwc = Lmwc

        # HESSIAN - SET ATTRIBUTE [ATOMIC UNITS AT THIS POINT]
        self.hessian = np.diagflat(rphval[3:])

        # FREQUENCIES - SET ATTRIBUTE
        self.frequencies = vib_freq_cm

        return vib_freq_cm, Lmwc


def driver_process_abinitio_data(version_gaussian, data_minima):
    """Driver function to process raw abinitio data obtained from GAUSSIAN

    Parameters
    ----------
    version_gaussian: str
        Gaussian version: 09 or 16
    data_minima: array_like
        List of output files with relativep path

    Returns
    -------
    object
        Molecule class object with new values

    """

    E0, ZPE = read_log_gaussian(data_minima[0])
    natom, anums, amass, cgeom, gradient, fcm = read_fchk_gaussian(data_minima[1], version_gaussian)

    # create instance of Molecule
    object_minima = Molecule(natom, amass, cgeom, fcm)

    # Get the matrix of atomic masses
    mass_matrix_sqrt_div = np.diag(np.repeat(1.0 / np.sqrt(amass), 3))

    # Mass-weigthed carteisan gradient (amu^-0.5*eV)
    gradient_mw = np.dot(mass_matrix_sqrt_div, gradient * au_to_ev)

    # set attribute for energy and zpe
    setattr(object_minima, 'energy', E0)
    setattr(object_minima, 'zpe', ZPE)

    # mass-weighted cartesian coordinates
    object_minima.get_mass_weighted_cartesian_coordinates()

    # centre of mass
    object_minima.get_centre_of_mass_mwc()

    # moment of inertia tensor
    object_minima.get_inertia_tensor()

    # normal coordinates [HARMONIC VIBRATIONAL ANALYSIS]
    old_vib_freq_cm, object_minima_lmwc = object_minima.harmonic_vibrational_analysis

    # Test if normal modes are orthonormal
    if np.allclose(np.dot(object_minima_lmwc.T, object_minima_lmwc), np.identity(object_minima.nmode)):
        pass
    else:
        raise ValueError('Minima normal modes not orthonormal. Exiting...')

    # project out rotations and translations from mass-weighted gradient
    LTR = object_minima.Ltransrot
    Imat = np.eye(LTR.shape[0])
    llt = np.dot(LTR, LTR.T)
    gradient_mw = np.dot(gradient_mw.T, (Imat - llt))

    # transform gradient to normal mode coordinates
    gradient_nm = np.dot(object_minima_lmwc.T, gradient_mw)
    setattr(object_minima, 'gradient', gradient_nm)

    # convert hessian to new units [eV / amu * Bohr^2]
    hessian_new = getattr(object_minima, 'hessian') * au_to_ev
    setattr(object_minima, 'hessian', hessian_new)

    return object_minima
