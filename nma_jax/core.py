"""Pure-JAX numerical core for normal-mode analysis.

All functions in this module are pure: they take arrays in, return arrays out,
and have no side effects.  Where shapes are statically known they are JIT-able.
The higher-level ``Molecule`` wrapper in :mod:`nma.molecule` orchestrates these.

Conventions
-----------
* Coordinates ``coords`` have shape ``(N, 3)`` and are in **Bohr** (atomic units).
* Masses ``masses`` have shape ``(N,)`` and are in **atomic mass units (amu)**.
* The Cartesian Hessian ``hessian`` has shape ``(3N, 3N)`` and is in
  **Hartree / Bohr^2** (atomic units of force constant).
* Frequencies are returned in **cm^-1**.
* Imaginary frequencies are reported as negative real numbers (Gaussian convention).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .constants import HFREQ_CM

# ----------------------------------------------------------------------
# Geometric properties
# ----------------------------------------------------------------------


@jax.jit
def center_of_mass(masses: Array, coords: Array) -> Array:
    """Return the centre-of-mass vector ``(3,)`` for a molecule.

    Parameters
    ----------
    masses : (N,) array
        Atomic masses (amu).
    coords : (N, 3) array
        Cartesian coordinates (Bohr).
    """
    return (masses[:, None] * coords).sum(axis=0) / masses.sum()


@jax.jit
def shift_to_com(masses: Array, coords: Array) -> Array:
    """Return ``coords`` translated so that the COM is at the origin."""
    return coords - center_of_mass(masses, coords)


@jax.jit
def inertia_tensor(masses: Array, coords: Array) -> Array:
    """Compute the moment-of-inertia tensor in **standard physics convention**.

    .. math::

        I_{\\alpha\\alpha} = \\sum_a m_a (r_a^2 - r_{a,\\alpha}^2)
        \\qquad
        I_{\\alpha\\beta} = -\\sum_a m_a r_{a,\\alpha} r_{a,\\beta}

    The off-diagonal sign convention is the **correct** one (a sign bug in the
    original Python-2 code mistakenly used ``+`` here, giving incorrect
    principal axes for molecules not already aligned).

    Parameters
    ----------
    masses : (N,) array
    coords : (N, 3) array
        Coordinates **must be COM-centered** for the resulting tensor to have
        physical meaning as a body-frame inertia tensor.

    Returns
    -------
    (3, 3) array
    """
    # I_ij = delta_ij * sum_a m_a |r_a|^2 - sum_a m_a r_a,i r_a,j
    r2 = (coords ** 2).sum(axis=1)                    # (N,)
    trace_term = (masses * r2).sum() * jnp.eye(3)     # (3, 3)
    outer = jnp.einsum("a,ai,aj->ij", masses, coords, coords)
    return trace_term - outer


@jax.jit
def principal_axes(masses: Array, coords: Array) -> tuple[Array, Array]:
    """Return ``(I_principal, axes)`` from the inertia tensor.

    ``axes`` is a ``(3, 3)`` matrix whose **columns** are the principal axes
    sorted by ascending moment of inertia; ``I_principal`` are the eigenvalues.
    """
    com_coords = shift_to_com(masses, coords)
    eigvals, eigvecs = jnp.linalg.eigh(inertia_tensor(masses, com_coords))
    return eigvals, eigvecs


def is_linear(
    masses: Array,
    coords: Array,
    *,
    rtol: float = 1e-4,
) -> bool:
    """Heuristic test for a linear molecule.

    A molecule is considered linear when the smallest principal moment of
    inertia is essentially zero compared to the largest.  Atomic molecules
    (N=1) and diatomics (N=2) are always linear.
    """
    n = masses.shape[0]
    if n <= 2:
        return True
    moi, _ = principal_axes(masses, coords)
    moi = jnp.sort(moi)
    return bool(moi[0] < rtol * moi[-1])


# ----------------------------------------------------------------------
# Translation / rotation projectors
# ----------------------------------------------------------------------


@jax.jit
def translation_vectors(masses: Array) -> Array:
    """Mass-weighted infinitesimal-translation vectors, shape ``(3N, 3)``.

    Column ``i`` is a unit vector that displaces every atom by the same amount
    along Cartesian direction ``i`` (mass-weighted, then orthonormalised here
    by construction since the columns are already orthogonal).
    """
    sqrt_m = jnp.sqrt(masses)
    # T[3k + i, j] = sqrt(m_k) * delta_{ij}
    # einsum: ('a,ij->aij') gives shape (N,3,3); reshape to (3N,3)
    raw = jnp.einsum("a,ij->aij", sqrt_m, jnp.eye(3)).reshape(3 * masses.shape[0], 3)
    # Normalise each of the 3 columns (||T_i||^2 = sum_k m_k = M_total)
    norm = jnp.sqrt((raw ** 2).sum(axis=0, keepdims=True))
    return raw / norm


def rotation_vectors(masses: Array, coords: Array) -> Array:
    """Mass-weighted infinitesimal-rotation vectors.

    Returns
    -------
    (3N, 3) array for a non-linear molecule, ``(3N, 2)`` for a linear one.

    Notes
    -----
    Uses the Sayvetz construction in the principal-axis frame
    (Wilson, Decius & Cross, *Molecular Vibrations*, §II-6).
    """
    n = masses.shape[0]
    com_coords = shift_to_com(masses, coords)
    moi, axes = jnp.linalg.eigh(inertia_tensor(masses, com_coords))

    # Coordinates expressed in the principal-axis frame: P[i, k] = (R_i · axes[:, k])
    p = com_coords @ axes                                 # (N, 3)
    sqrt_m = jnp.sqrt(masses)

    # D_rot[3i + j, alpha] = (P[i, gamma]*axes[j, beta] - P[i, beta]*axes[j, gamma]) * sqrt(m_i)
    # where (beta, gamma) cycles as (1,2), (2,0), (0,1) for alpha = 0, 1, 2.
    beta = jnp.array([1, 2, 0])
    gamma = jnp.array([2, 0, 1])
    pg = p[:, gamma]                                      # (N, 3)   P[i, gamma[alpha]]
    pb = p[:, beta]                                       # (N, 3)   P[i, beta[alpha]]
    ab = axes[:, beta]                                    # (3, 3)   axes[j, beta[alpha]]
    ag = axes[:, gamma]                                   # (3, 3)   axes[j, gamma[alpha]]
    # Shape (N, 3, 3) -> (atom i, cartesian j, rotation alpha)
    d = (pg[:, None, :] * ab[None, :, :] - pb[:, None, :] * ag[None, :, :])
    d_rot = (d * sqrt_m[:, None, None]).reshape(3 * n, 3)  # (3N, 3)

    # For linear molecules, one column has near-zero norm and must be dropped
    norms = jnp.linalg.norm(d_rot, axis=0)
    if bool((norms < 1e-8 * norms.max()).any()):
        keep = jnp.argsort(-norms)[:2]
        d_rot = d_rot[:, keep]

    # Orthonormalise (QR is more numerically reliable than dividing by norm
    # because the three rotation modes are not mutually orthogonal for a
    # general inertia tensor).
    q, _ = jnp.linalg.qr(d_rot)
    return q


def project_out_trans_rot(
    masses: Array,
    coords: Array,
    matrix: Array,
    *,
    project_translation: bool = True,
    project_rotation: bool = True,
) -> tuple[Array, Array]:
    """Project translations/rotations out of a mass-weighted ``(3N, 3N)`` matrix.

    Parameters
    ----------
    masses, coords
        Atomic masses and coordinates.
    matrix
        Mass-weighted Hessian or similar ``(3N, 3N)`` operator.
    project_translation, project_rotation
        Toggle each projection.

    Returns
    -------
    projected : (3N, 3N)
        ``(I - P) @ matrix @ (I - P)``.
    basis : (3N, k)
        Concatenation of the orthonormal translation+rotation basis vectors
        that were projected out (k = 0, 3, 5, or 6 depending on molecule
        geometry and which projections are requested).
    """
    n = masses.shape[0]
    pieces: list[Array] = []
    if project_translation:
        pieces.append(translation_vectors(masses))
    if project_rotation:
        pieces.append(rotation_vectors(masses, coords))

    if not pieces:
        return matrix, jnp.zeros((3 * n, 0))

    basis = jnp.concatenate(pieces, axis=1)
    # Final orthonormalisation to safeguard against linear dependence between
    # translation and rotation columns (which is exact in exact arithmetic
    # but slightly off in floating point).
    basis, _ = jnp.linalg.qr(basis)
    p = basis @ basis.T
    eye = jnp.eye(3 * n)
    return (eye - p) @ matrix @ (eye - p), basis


# ----------------------------------------------------------------------
# Mass-weighted Hessian and frequencies
# ----------------------------------------------------------------------


@jax.jit
def mass_weight_hessian(hessian: Array, masses: Array) -> Array:
    r"""Return ``M^{-1/2} H M^{-1/2}`` for atomic Hessian ``hessian``.

    The mass matrix :math:`M` is diagonal with each atomic mass repeated
    three times (one per Cartesian direction).
    """
    inv_sqrt_m3 = 1.0 / jnp.sqrt(jnp.repeat(masses, 3))    # (3N,)
    return hessian * inv_sqrt_m3[:, None] * inv_sqrt_m3[None, :]


def eigenvalues_to_frequencies(eigvals: Array) -> Array:
    """Convert mass-weighted-Hessian eigenvalues (au) to wavenumbers (cm^-1).

    Imaginary modes (negative eigenvalues) are returned as **negative** real
    numbers, matching Gaussian's convention for transition states.
    """
    sign = jnp.sign(eigvals)
    return sign * jnp.sqrt(jnp.abs(eigvals)) * HFREQ_CM


# ----------------------------------------------------------------------
# Top-level normal-mode analysis
# ----------------------------------------------------------------------


def harmonic_analysis(
    masses: Array,
    coords: Array,
    hessian: Array,
    *,
    project_translation: bool = True,
    project_rotation: bool = True,
    linear: bool | None = None,
) -> dict[str, Array]:
    """Diagonalise the mass-weighted Hessian after projecting out trans/rot.

    Parameters
    ----------
    masses : (N,) array (amu)
    coords : (N, 3) array (Bohr)
    hessian : (3N, 3N) array (Hartree / Bohr^2)
    project_translation, project_rotation
        Toggle each projection (default: both on).
    linear
        If ``None`` (default), determined automatically.

    Returns
    -------
    dict
        ``frequencies`` (cm^-1), shape ``(nmode,)`` with imaginary modes
        negative.

        ``modes_mwc`` (3N, nmode) mass-weighted Cartesian normal coordinates
        (columns are unit-norm).

        ``modes_cart`` (3N, nmode) Cartesian displacements ``M^{-1/2} L_mwc``,
        useful for visualisation.

        ``hessian_mwc_diag`` diagonal Hessian in the normal-mode basis (au).

        ``projector_basis`` orthonormal columns spanning the projected-out
        subspace.

        ``nmode`` number of vibrational modes returned.
    """
    n = masses.shape[0]
    if linear is None:
        linear = is_linear(masses, coords)

    # Expected count of zero modes
    n_trans = 3 if project_translation else 0
    n_rot = (0 if not project_rotation else (2 if linear else 3))
    n_zero = n_trans + n_rot
    n_mode = 3 * n - n_zero

    h_mw = mass_weight_hessian(hessian, masses)
    h_proj, basis = project_out_trans_rot(
        masses, coords, h_mw,
        project_translation=project_translation,
        project_rotation=project_rotation,
    )

    eigvals, eigvecs = jnp.linalg.eigh(h_proj)

    # eigh sorts ascending.  Zero modes have eigenvalues ~ 0 and can appear
    # at the start (or scattered with tiny negative values).  Identify them
    # by absolute magnitude and move them to the front so they slice off
    # cleanly.
    order = jnp.argsort(jnp.abs(eigvals))
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]

    # Zero modes are the n_zero smallest-magnitude eigenvalues; the rest are vibrational
    zero_vals = eigvals_sorted[:n_zero]
    zero_vecs = eigvecs_sorted[:, :n_zero]
    vib_vals = eigvals_sorted[n_zero:]
    vib_vecs = eigvecs_sorted[:, n_zero:]

    # Re-sort vibrational modes by ascending eigenvalue (puts imaginary first)
    order_v = jnp.argsort(vib_vals)
    vib_vals = vib_vals[order_v]
    vib_vecs = vib_vecs[:, order_v]

    freqs = eigenvalues_to_frequencies(vib_vals)

    # Cartesian normal modes for visualisation: M^{-1/2} * L_mwc
    inv_sqrt_m3 = 1.0 / jnp.sqrt(jnp.repeat(masses, 3))
    modes_cart = inv_sqrt_m3[:, None] * vib_vecs

    return {
        "frequencies": freqs,
        "modes_mwc": vib_vecs,
        "modes_cart": modes_cart,
        "eigenvalues_mwc": vib_vals,
        "zero_mode_eigenvalues": zero_vals,
        "zero_mode_vectors": zero_vecs,
        "projector_basis": basis,
        "nmode": n_mode,
        "is_linear": linear,
    }
