"""Harmonic vibrational analysis in mass-weighted Cartesian coordinates.

This module is intentionally **pure**: every function takes arrays and returns
arrays, with no hidden state. The expensive routines are jit-friendly.

The projection of translations and rotations is done by constructing
infinitesimal generators directly (cf. NWChem / Wilson, Decius & Cross,
Section 11-2). This is more numerically robust than the principal-axis-frame
trick used in the original code and trivially extends to linear molecules.

Imaginary frequencies are returned as **negative wavenumbers** by convention:
this matches Gaussian's printed output for transition states and lets you
distinguish them at a glance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .constants import FREQ_AU_TO_CM
from .molecule import Molecule

Array = jax.Array

# Cutoff (in atomic units of eigenvalue) below which a mode is considered
# spurious (translation / rotation). The eigenvalue is in Hartree / amu / Bohr^2,
# corresponding to roughly sqrt(1e-10) * 5140 ~ 0.05 cm^-1 when freq-converted —
# safely below any physical mode.
_ZERO_EIG_TOL = 1e-10
# Cutoff for detecting linear molecules (relative to the largest moment).
_LINEAR_TOL = 1e-6


# -----------------------------------------------------------------------------
# Public result type
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class VibrationalAnalysis:
    """Result of a harmonic vibrational analysis.

    Attributes
    ----------
    frequencies_cm : Array of shape (nmode,)
        Harmonic frequencies in cm^-1. **Imaginary frequencies are returned
        as negative numbers** (the standard Gaussian convention).
    modes_mw_cart : Array of shape (3*natom, nmode)
        Mass-weighted Cartesian normal modes (the eigenvectors of the
        projected mass-weighted Hessian). Columns are orthonormal.
    modes_cart : Array of shape (3*natom, nmode)
        Cartesian normal modes (i.e. M^{-1/2} times the mass-weighted modes).
        These are the actual atomic displacement patterns; columns are
        **not** orthonormal in general.
    reduced_masses : Array of shape (nmode,), in amu
        :math:`\\mu_k = 1 / \\sum_i |L^{cart}_{ik}|^2`.
    eigenvalues_au : Array of shape (nmode,)
        Eigenvalues of the projected mass-weighted Hessian (Hartree / amu / Bohr^2).
        Negative values correspond to imaginary frequencies.
    projector_tr : Array of shape (3*natom, n_tr)
        Orthonormal basis for the translation/rotation subspace that was
        projected out (5 columns for linear molecules, 6 otherwise — or
        fewer if you disabled translation/rotation projection).
    is_linear : bool
        Whether the molecule was detected as linear.
    nmode : int
    """

    frequencies_cm: Array
    modes_mw_cart: Array
    modes_cart: Array
    reduced_masses: Array
    eigenvalues_au: Array
    projector_tr: Array
    is_linear: bool
    nmode: int

    def __repr__(self) -> str:
        n_imag = int(jnp.sum(self.frequencies_cm < 0))
        return (
            f"VibrationalAnalysis(nmode={self.nmode}, "
            f"n_imaginary={n_imag}, linear={self.is_linear})"
        )


# -----------------------------------------------------------------------------
# Building blocks (jittable)
# -----------------------------------------------------------------------------
def mass_weighting_matrix(masses: Array) -> Array:
    """Return diag(1/sqrt(m_i)) repeated 3x: shape (3N, 3N)."""
    return jnp.diag(jnp.repeat(1.0 / jnp.sqrt(masses), 3))


def inertia_tensor(coordinates: Array, masses: Array) -> Array:
    """Inertia tensor in the standard physics convention (off-diagonals negative).

    I_ab = sum_i m_i (|r_i|^2 delta_ab - r_{i,a} r_{i,b})

    ``coordinates`` should already be centred on the COM for the eigenvectors
    to represent rotations about the COM.
    """
    r = coordinates
    m = masses
    r2 = jnp.sum(r * r, axis=1)
    I = jnp.eye(3) * jnp.sum(m * r2)
    I = I - jnp.einsum("i,ia,ib->ab", m, r, r)
    return I


def _is_linear(coordinates: Array, masses: Array) -> bool:
    """Detect a linear molecule by checking whether the smallest principal
    moment of inertia is negligible.
    """
    coords_c = coordinates - jnp.einsum("i,ij->j", masses, coordinates) / jnp.sum(masses)
    I = inertia_tensor(coords_c, masses)
    eigvals = jnp.linalg.eigvalsh(I)
    return bool(eigvals[0] < _LINEAR_TOL * jnp.maximum(eigvals[-1], 1.0))


def translation_rotation_basis(
    coordinates: Array,
    masses: Array,
    *,
    project_translation: bool = True,
    project_rotation: bool = True,
) -> tuple[Array, bool]:
    """Build an orthonormal basis for the translation/rotation subspace
    in mass-weighted Cartesian coordinates.

    Returns
    -------
    D : Array of shape (3*natom, n_tr)
        Orthonormal columns spanning the requested subspace.
    is_linear : bool

    Notes
    -----
    Translations come from the displacement of all atoms along x, y, z scaled
    by sqrt(m_i). Rotations come from the infinitesimal cross product
    sqrt(m_i) (e_alpha x r_i). For a linear molecule one rotational generator
    is linearly dependent and is dropped by QR.
    """
    natom = coordinates.shape[0]
    masses = jnp.asarray(masses)
    sqrt_m = jnp.sqrt(masses)

    # Centre coordinates on COM for rotation generators.
    com = jnp.einsum("i,ij->j", masses, coordinates) / jnp.sum(masses)
    r = coordinates - com  # (natom, 3)

    columns: list[Array] = []

    if project_translation:
        # T_alpha: sqrt(m_i) e_alpha for each atom
        for alpha in range(3):
            v = jnp.zeros((natom, 3))
            v = v.at[:, alpha].set(sqrt_m)
            columns.append(v.reshape(-1))

    if project_rotation:
        # R_alpha: sqrt(m_i) (e_alpha x r_i)
        # cross product: (e_x x r) = (0, -z, y), etc.
        x, y, z = r[:, 0], r[:, 1], r[:, 2]
        rx = jnp.stack([jnp.zeros_like(x), -z, y], axis=1)
        ry = jnp.stack([z, jnp.zeros_like(x), -x], axis=1)
        rz = jnp.stack([-y, x, jnp.zeros_like(x)], axis=1)
        for v in (rx, ry, rz):
            v = v * sqrt_m[:, None]
            columns.append(v.reshape(-1))

    D_raw = jnp.stack(columns, axis=1)  # (3N, ncols)

    # Detect linearity (we'll trim a column if needed).
    is_linear = _is_linear(coordinates, masses) if project_rotation else False

    # QR with column-pivoting would be ideal for dropping the redundant column
    # (e.g. one rotation generator for linear molecules); JAX doesn't expose
    # pivoted QR, so use SVD-based orthonormalisation — any column with
    # negligible singular value is dropped.
    U, s, _ = jnp.linalg.svd(D_raw, full_matrices=False)
    smax = jnp.maximum(s[0], 1e-30)
    keep_mask = jnp.asarray(s > 1e-8 * smax)
    import numpy as _np
    keep_idx = _np.where(_np.asarray(keep_mask))[0]
    D = U[:, keep_idx]

    return D, is_linear


def project_out(matrix: Array, basis: Array) -> Array:
    """Project ``matrix`` onto the orthogonal complement of ``basis``.

    Both projector applications: P A P with P = I - B B^T.
    """
    n = matrix.shape[0]
    P = jnp.eye(n) - basis @ basis.T
    return P @ matrix @ P


# -----------------------------------------------------------------------------
# The main analysis routine
# -----------------------------------------------------------------------------
def harmonic_analysis(
    molecule: Molecule,
    *,
    project_translation: bool = True,
    project_rotation: bool = True,
) -> VibrationalAnalysis:
    """Run a complete harmonic vibrational analysis.

    Parameters
    ----------
    molecule : Molecule
        Coordinates in Bohr, masses in amu, Hessian in Hartree / Bohr^2.
    project_translation, project_rotation : bool
        Whether to project out overall translations / rotations.

    Returns
    -------
    VibrationalAnalysis
    """
    coords = molecule.coordinates
    masses = molecule.masses
    H = molecule.hessian_cart

    # Mass-weighted Hessian:   F = M^{-1/2} H M^{-1/2}
    Mhalf_inv = mass_weighting_matrix(masses)
    F = Mhalf_inv @ H @ Mhalf_inv

    # Build translation/rotation basis & project out
    D_tr, is_linear = translation_rotation_basis(
        coords, masses,
        project_translation=project_translation,
        project_rotation=project_rotation,
    )
    n_tr = D_tr.shape[1]
    F_proj = project_out(F, D_tr)

    # Diagonalise
    eigvals, eigvecs = jnp.linalg.eigh(F_proj)

    # The translation/rotation directions return eigenvalues ~ 0 (because we
    # projected them out). Identify and remove them by absolute magnitude.
    keep_mask = jnp.abs(eigvals) > _ZERO_EIG_TOL
    import numpy as _np
    keep_idx = _np.where(_np.asarray(keep_mask))[0]
    # Sanity: we should be left with exactly ndof - n_tr modes.
    nmode_expected = molecule.ndof - n_tr
    if keep_idx.size != nmode_expected:
        # Fall back to taking the n_tr smallest-by-|eigenvalue| modes as the
        # spurious ones — robust if the tolerance was slightly off.
        order = _np.argsort(_np.abs(_np.asarray(eigvals)))
        keep_idx = _np.sort(order[n_tr:])

    eigvals = eigvals[keep_idx]
    eigvecs = eigvecs[:, keep_idx]

    # Sort vibrational modes by signed eigenvalue (imaginary first, then
    # ascending real). This is convenient for TS analysis.
    order = jnp.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Convert to wavenumbers, keeping sign:
    freq_cm = jnp.sign(eigvals) * jnp.sqrt(jnp.abs(eigvals)) * FREQ_AU_TO_CM

    # Cartesian (unweighted) normal modes:  L_cart = M^{-1/2} L_mwc
    L_cart = Mhalf_inv @ eigvecs

    # Reduced masses (amu): mu_k = 1 / sum_i |L^cart_{i,k}|^2 if L_mwc is
    # orthonormal. Equivalent shorter form below.
    norms2 = jnp.sum(L_cart * L_cart, axis=0)
    reduced_masses = 1.0 / jnp.maximum(norms2, 1e-30)

    return VibrationalAnalysis(
        frequencies_cm=freq_cm,
        modes_mw_cart=eigvecs,
        modes_cart=L_cart,
        reduced_masses=reduced_masses,
        eigenvalues_au=eigvals,
        projector_tr=D_tr,
        is_linear=bool(is_linear),
        nmode=int(eigvals.shape[0]),
    )


def project_gradient_to_modes(
    molecule: Molecule,
    analysis: VibrationalAnalysis,
) -> Array:
    """Transform a Cartesian gradient into the normal-mode basis.

    Returns an array of shape ``(nmode,)`` in Hartree / (sqrt(amu) Bohr).
    """
    if molecule.gradient_cart is None:
        raise ValueError("Molecule has no Cartesian gradient set.")
    Mhalf_inv = mass_weighting_matrix(molecule.masses)
    g_mw = Mhalf_inv @ molecule.gradient_cart            # (3N,)
    # Project out translation/rotation directions for consistency
    P = jnp.eye(g_mw.shape[0]) - analysis.projector_tr @ analysis.projector_tr.T
    g_mw = P @ g_mw
    return analysis.modes_mw_cart.T @ g_mw


__all__ = [
    "VibrationalAnalysis",
    "harmonic_analysis",
    "translation_rotation_basis",
    "inertia_tensor",
    "mass_weighting_matrix",
    "project_out",
    "project_gradient_to_modes",
]
