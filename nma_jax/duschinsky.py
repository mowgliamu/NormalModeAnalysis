"""Duschinsky rotation and Eckart alignment between two molecules.

The Duschinsky transformation relates the normal modes of two electronic
states (or two minima, or a minimum and a transition state) of the same
molecule:

.. math::
    \\mathbf{Q}' = \\mathbf{J} \\mathbf{Q} + \\mathbf{K}

where :math:`\\mathbf{J}` is the Duschinsky rotation matrix and
:math:`\\mathbf{K}` is the dimensionless displacement vector along the
modes of the target state. The two structures must first be put in a
common (Eckart) frame to make this geometrically meaningful.

References
----------
- F. Duschinsky, Acta Physicochim. URSS **7**, 551 (1937)
- Sharp & Rosenstock, JCP **41**, 3453 (1964)
- Reimers, JCP **115**, 9103 (2001)
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .constants import DIMLESS_PREFACTOR
from .molecule import Molecule
from .vibrational import VibrationalAnalysis, mass_weighting_matrix

Array = jax.Array


@dataclass(frozen=True)
class DuschinskyResult:
    """Result of a Duschinsky analysis.

    Attributes
    ----------
    J : (nmode, nmode)  — Duschinsky rotation matrix in the mass-weighted basis
    K : (nmode,)        — displacement (sqrt(amu) * Bohr), in target's normal modes
    K_dimensionless : (nmode,)  — same displacement scaled by sqrt(omega'/hbar)
    eckart_rotation : (3, 3)  — rotation applied to ``mol_source`` to align with target
    aligned_source_coords : (natom, 3)  — source coordinates after Eckart alignment
    rms_alignment : float — RMS atom-by-atom mass-weighted displacement after alignment
    """
    J: Array
    K: Array
    K_dimensionless: Array
    eckart_rotation: Array
    aligned_source_coords: Array
    rms_alignment: float


def eckart_rotation_matrix(
    source_coords: Array,
    target_coords: Array,
    masses: Array,
) -> Array:
    """Find the rotation that minimises mass-weighted RMS deviation
    of ``source`` from ``target``.

    Both inputs should already be COM-centred. Uses the Kabsch / Wahba
    algorithm with a determinant correction so we never return a reflection.

    Returns
    -------
    R : (3, 3) proper rotation matrix
    """
    # Cross-covariance C_ij = sum_a m_a target_{a,i} source_{a,j}
    C = jnp.einsum("a,ai,aj->ij", masses, target_coords, source_coords)
    U, _, Vt = jnp.linalg.svd(C)
    # Determinant correction (avoid reflection)
    d = jnp.sign(jnp.linalg.det(U @ Vt))
    D = jnp.diag(jnp.array([1.0, 1.0, d]))
    R = U @ D @ Vt
    return R


def _com_center(coords: Array, masses: Array) -> Array:
    com = jnp.einsum("a,ai->i", masses, coords) / jnp.sum(masses)
    return coords - com


def duschinsky(
    mol_source: Molecule,
    analysis_source: VibrationalAnalysis,
    mol_target: Molecule,
    analysis_target: VibrationalAnalysis,
) -> DuschinskyResult:
    """Compute the Duschinsky matrix J and displacement K relating
    two molecular structures of the same molecule.

    The source structure is rotated/translated into the Eckart frame
    of the target before computing J and K. Both vibrational analyses
    must have the same number of modes (i.e. the same atoms, same
    projection choices).

    Parameters
    ----------
    mol_source, mol_target : Molecule
        Source (initial) and target (final) molecules. Same atom ordering!
    analysis_source, analysis_target : VibrationalAnalysis
        Their vibrational analyses, in mass-weighted Cartesian coordinates.

    Returns
    -------
    DuschinskyResult
    """
    if mol_source.natom != mol_target.natom:
        raise ValueError("Source and target must have the same number of atoms.")
    if analysis_source.nmode != analysis_target.nmode:
        raise ValueError("Source and target must have the same number of modes.")
    if not bool(jnp.all(mol_source.atomic_numbers == mol_target.atomic_numbers)):
        raise ValueError("Source and target must have the same atom ordering.")

    masses = mol_target.masses  # we assume both have the same masses

    # 1. COM-centre both
    src_c = _com_center(mol_source.coordinates, masses)
    tgt_c = _com_center(mol_target.coordinates, masses)

    # 2. Eckart rotation: rotate source into target's frame
    R = eckart_rotation_matrix(src_c, tgt_c, masses)
    src_aligned = src_c @ R.T

    # 3. Mass-weighted displacement vector (sqrt(amu) * Bohr)
    sqrt_m = jnp.sqrt(masses)[:, None]
    delta_mw = (sqrt_m * (src_aligned - tgt_c)).reshape(-1)  # (3N,) — source minus target

    # The displacement in the target's normal-mode basis:
    K = analysis_target.modes_mw_cart.T @ delta_mw  # (nmode,)

    # 4. Duschinsky matrix: rotate the source's mass-weighted modes into the
    #    target frame, then express them in the target's normal mode basis.
    #    L_src is (3N, nmode) where columns are mass-weighted modes.
    #    The rotation R acts on the spatial (3,) part of each atom: build a
    #    block-diagonal rotation in 3N.
    natom = mol_source.natom
    R_block = jnp.kron(jnp.eye(natom), R)  # (3N, 3N)
    L_src_rotated = R_block @ analysis_source.modes_mw_cart
    J = analysis_target.modes_mw_cart.T @ L_src_rotated  # (nmode, nmode)

    # 5. Dimensionless displacement K~ = sqrt(omega/hbar) K
    eig = analysis_target.eigenvalues_au
    sqrt_omega = jnp.where(eig > 0, jnp.sqrt(jnp.abs(eig)), 0.0)
    scale = DIMLESS_PREFACTOR * jnp.sqrt(sqrt_omega)
    K_dim = scale * K

    rms = float(jnp.sqrt(jnp.mean(delta_mw**2)))

    return DuschinskyResult(
        J=J,
        K=K,
        K_dimensionless=K_dim,
        eckart_rotation=R,
        aligned_source_coords=src_aligned,
        rms_alignment=rms,
    )


__all__ = ["DuschinskyResult", "eckart_rotation_matrix", "duschinsky"]
