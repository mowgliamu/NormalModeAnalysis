r"""Eckart alignment between two geometries and Duschinsky-rotation analysis.

These are the operations declared as attributes (``EckartMatrix``,
``Duschinsky``, ``Displacement``) in the original code but never actually
implemented.  They are essential for inelastic-scattering rates, fluorescence
spectra (FCFs), and any analysis that compares two molecular geometries
through their normal-mode bases.

Algorithm: mass-weighted Kabsch--Umeyama
----------------------------------------
Given two COM-centred coordinate sets :math:`R_a` (reference) and
:math:`R_a^\prime` (displaced) with masses :math:`m_a`, the optimal proper
rotation :math:`U` minimising

.. math::

    \\Phi(U) = \\sum_a m_a \\lVert R_a^\\prime - U R_a \\rVert^2

is obtained from the SVD

.. math::

    C = \\sum_a m_a R_a^\\prime R_a^T = V \\Sigma W^T,
    \\qquad U = V \\operatorname{diag}(1, 1, \\det(V W^T)) W^T.

The ``diag(1, 1, det(.))`` correction guarantees a proper rotation (no
reflection), following Kabsch (1976) / Umeyama (1991).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from . import core
from .molecule import Molecule, NormalModes


# ----------------------------------------------------------------------
# Eckart rotation matrix
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EckartResult:
    """Result of mass-weighted Eckart alignment.

    Attributes
    ----------
    rotation : (3, 3) Array
        Proper rotation matrix that, when applied as ``coords_ref @ rotation``
        from the right, gives the best alignment to ``coords_disp``.
    rmsd : float
        Mass-weighted RMSD between the rotated reference and the displaced
        geometry (Bohr).
    """
    rotation: Array
    rmsd: float


@jax.jit
def _kabsch(masses: Array, ref: Array, disp: Array) -> Array:
    """Core: return the proper rotation ``U`` such that ``ref @ U ~= disp``.

    Conventions
    -----------
    ``ref`` and ``disp`` are ``(N, 3)`` arrays of row-vector atom coordinates,
    already COM-centred.  The returned matrix is the right-multiplying form
    (``coords @ U``), consistent with :func:`apply_eckart_rotation` and the
    rest of the package.

    Internally this is the standard mass-weighted Kabsch--Umeyama SVD on the
    cross-covariance ``C = sum_a m_a r'_a r_a^T``.  Standard Kabsch yields the
    column-form rotation ``R = V diag(1, 1, det(VW^T)) W^T`` (so that
    ``r'_a = R r_a``); we return its transpose to match the row convention.
    """
    # ref, disp already COM-centred
    c = jnp.einsum("a,ai,aj->ij", masses, disp, ref)        # (3, 3)
    v, _s, wt = jnp.linalg.svd(c)
    sign = jnp.sign(jnp.linalg.det(v @ wt))
    d = jnp.diag(jnp.array([1.0, 1.0, sign]))
    r_col = v @ d @ wt
    return r_col.T


def eckart_align(reference: Molecule, displaced: Molecule) -> EckartResult:
    """Return the proper rotation that brings ``reference`` onto ``displaced``.

    Both molecules must have the same atoms in the same order; masses are
    taken from ``reference``.

    The returned rotation ``U`` is such that ``reference.coords @ U`` (after
    COM-centering each side) best matches ``displaced.coords``.
    """
    if reference.n_atoms != displaced.n_atoms:
        raise ValueError(
            f"Atom-count mismatch: {reference.n_atoms} vs {displaced.n_atoms}"
        )
    ref_c = core.shift_to_com(reference.masses, reference.coords)
    disp_c = core.shift_to_com(reference.masses, displaced.coords)
    u = _kabsch(reference.masses, ref_c, disp_c)
    # Mass-weighted RMSD after alignment
    rotated = ref_c @ u
    diff = disp_c - rotated
    msd = (reference.masses[:, None] * diff ** 2).sum() / reference.masses.sum()
    return EckartResult(rotation=u, rmsd=float(jnp.sqrt(msd)))


def apply_eckart_rotation(mol: Molecule, rotation: Array) -> Molecule:
    """Return a new molecule with coordinates (and Hessian, if any) rotated.

    The rotation is applied as ``coords @ rotation`` (right multiplication).
    The Hessian, if present, is transformed as ``H' = R_{3N}^T H R_{3N}``
    where :math:`R_{3N}` is block-diagonal with ``rotation`` on each block.
    """
    new_coords = mol.coords @ rotation
    new_h = mol.hessian
    if new_h is not None:
        r3n = jax.scipy.linalg.block_diag(*([rotation] * mol.n_atoms))
        new_h = r3n.T @ new_h @ r3n
    from dataclasses import replace
    return replace(mol, coords=new_coords, hessian=new_h)


# ----------------------------------------------------------------------
# Duschinsky matrix and displacement
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DuschinskyResult:
    """Output of Duschinsky analysis between two states.

    The Duschinsky transformation relates the mass-weighted normal coordinates
    :math:`Q^\\prime` of the displaced state to those :math:`Q` of the
    reference state via

    .. math::

        Q^\\prime = J Q + K,

    where :math:`J` is the rotation matrix and :math:`K` the displacement
    vector along the displaced-state normal coordinates.

    Attributes
    ----------
    J : (M, M) Array
        Duschinsky rotation matrix; ``J[k, l] = L'^T M^{1/2} L``-style overlap
        (after Eckart alignment).
    K : (M,) Array
        Displacement vector along the **displaced**-state mass-weighted
        normal coordinates: ``K = L'^T M^{1/2} (X' - U X)`` where ``U`` is
        the Eckart rotation, ``X'`` is the displaced geometry, ``X`` the
        reference geometry, and ``L'`` the displaced-state mwc normal modes.
    eckart_rotation : (3, 3) Array
    """
    J: Array
    K: Array
    eckart_rotation: Array


def duschinsky(reference: NormalModes, displaced: NormalModes) -> DuschinskyResult:
    """Compute the Duschinsky rotation matrix and displacement vector.

    Both arguments are :class:`NormalModes` objects (typically from
    :meth:`Molecule.normal_modes`).  The associated molecules must share
    the same atoms in the same order.

    Returns
    -------
    DuschinskyResult
    """
    ref_mol = reference.molecule
    disp_mol = displaced.molecule
    if ref_mol.n_atoms != disp_mol.n_atoms:
        raise ValueError(
            "Reference and displaced molecules have different atom counts."
        )

    # Mass-weighted, COM-centred Cartesians
    ref_c = core.shift_to_com(ref_mol.masses, ref_mol.coords)
    disp_c = core.shift_to_com(ref_mol.masses, disp_mol.coords)

    # Eckart-align reference to displaced
    u = _kabsch(ref_mol.masses, ref_c, disp_c)
    ref_aligned = ref_c @ u

    # J = L'^T L   in mass-weighted Cartesian normal-mode basis
    # but each L was constructed in its molecule's own (possibly differently
    # oriented) frame.  Rotate the reference L into the displaced frame:
    n = ref_mol.n_atoms
    r3n = jax.scipy.linalg.block_diag(*([u] * n))
    l_ref_rotated = r3n.T @ reference.modes_mwc          # express in displaced frame

    j = displaced.modes_mwc.T @ l_ref_rotated

    # K: displacement of equilibrium geometry projected onto displaced normal modes
    sqrt_m3 = jnp.sqrt(jnp.repeat(ref_mol.masses, 3))
    delta = (disp_c - ref_aligned).reshape(-1)            # (3N,) in Bohr
    delta_mw = sqrt_m3 * delta                            # mass-weighted
    k = displaced.modes_mwc.T @ delta_mw

    return DuschinskyResult(J=j, K=k, eckart_rotation=u)
