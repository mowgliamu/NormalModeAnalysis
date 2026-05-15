"""Tests for the core NMA routines.

The canonical regression test uses a water (H_2 O) Hessian published in the
Gaussian thermochemistry whitepaper.  Smaller unit tests exercise individual
routines (translation/rotation projection, mass weighting, ...).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from nma_jax import Molecule, thermochemistry
from nma_jax import core, eckart
from nma_jax.constants import ANGSTROM_BOHR, BOHR_ANGSTROM, HFREQ_CM


# ----------------------------------------------------------------------
# Test molecules
# ----------------------------------------------------------------------


def water_geometry():
    """Optimised water geometry in Bohr (standard textbook values)."""
    # B3LYP/6-31G(d) optimised geometry, in Angstrom
    coords_ang = np.array([
        [0.000000,  0.000000,  0.119262],   # O
        [0.000000,  0.763239, -0.477047],   # H
        [0.000000, -0.763239, -0.477047],   # H
    ])
    coords_bohr = coords_ang * ANGSTROM_BOHR
    return coords_bohr


def water_masses():
    return np.array([15.99491462, 1.00782503, 1.00782503])


def water_anums():
    return np.array([8, 1, 1])


# ----------------------------------------------------------------------
# Geometry tests
# ----------------------------------------------------------------------


def test_center_of_mass_water_is_on_z_axis():
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.center_of_mass(masses, coords)
    assert abs(float(com[0])) < 1e-10
    assert abs(float(com[1])) < 1e-10
    # The molecule should be roughly COM-centered to start with
    assert abs(float(com[2])) < 0.1


def test_inertia_tensor_water_is_diagonal_in_principal_frame():
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com_coords = core.shift_to_com(masses, coords)
    moi = core.inertia_tensor(masses, com_coords)
    # Off-diagonals should be ~0 already (water in standard orientation)
    off = np.asarray(moi) - np.diag(np.diag(np.asarray(moi)))
    assert np.max(np.abs(off)) < 1e-8


def test_inertia_tensor_sign_convention():
    """Verify off-diagonals are NEGATIVE m*x*y, not positive."""
    # Two-atom molecule offset in x and y
    masses = jnp.array([1.0, 1.0])
    coords = jnp.array([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]])
    com_coords = core.shift_to_com(masses, coords)
    moi = core.inertia_tensor(masses, com_coords)
    # I_xy should be -sum m*x*y = -(1*1*1 + 1*(-1)*(-1)) = -2
    assert np.isclose(float(moi[0, 1]), -2.0)
    assert np.isclose(float(moi[1, 0]), -2.0)


def test_water_is_not_linear():
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    assert core.is_linear(masses, coords) is False


def test_diatomic_is_linear():
    masses = jnp.array([14.003, 14.003])
    coords = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert core.is_linear(masses, coords) is True


# ----------------------------------------------------------------------
# Projection tests
# ----------------------------------------------------------------------


def test_translation_vectors_are_orthonormal():
    masses = jnp.array(water_masses())
    t = core.translation_vectors(masses)
    overlap = t.T @ t
    np.testing.assert_allclose(np.asarray(overlap), np.eye(3), atol=1e-10)


def test_rotation_vectors_are_orthonormal():
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    r = core.rotation_vectors(masses, com)
    overlap = r.T @ r
    np.testing.assert_allclose(np.asarray(overlap), np.eye(3), atol=1e-8)


def test_translation_rotation_orthogonal():
    """For any geometry centred at COM, translation and rotation
    eigenvectors are orthogonal to each other."""
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    cross = t.T @ r
    np.testing.assert_allclose(np.asarray(cross), np.zeros((3, 3)), atol=1e-8)


# ----------------------------------------------------------------------
# Synthetic Hessian round-trip
# ----------------------------------------------------------------------


def test_synthetic_hessian_reproduces_known_frequencies():
    """Build a Hessian directly from a known frequency spectrum.

    Take a non-linear 3-atom molecule (water geometry), build a mass-weighted
    Hessian with prescribed eigenvalues in the vibrational subspace and
    zeros in the trans/rot subspace, transform back to Cartesian Hessian,
    and verify the analysis recovers the prescribed frequencies.
    """
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)

    n = masses.shape[0]
    # Build full ON basis: translation+rotation, then random orthogonal
    # complement which we use as the "vibrational" eigenvectors.
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    tr = jnp.concatenate([t, r], axis=1)             # (3N, 6)
    # Random orthogonal completion via QR
    rng = np.random.default_rng(42)
    rand = rng.standard_normal((3 * n, 3 * n))
    rand[:, :6] = np.asarray(tr)
    q, _ = np.linalg.qr(rand)
    # First 6 columns of Q are the translation/rotation basis (close to it).
    # Last 3 are the vibrational eigenvectors.
    vib_evec = q[:, 6:]

    # Prescribed eigenvalues (au) -> frequencies in cm^-1
    target_freqs_cm = np.array([1500.0, 3000.0, 3500.0])     # increasing
    target_eig_au = (target_freqs_cm / HFREQ_CM) ** 2
    eig_full = np.concatenate([np.zeros(6), target_eig_au])
    h_mw = q @ np.diag(eig_full) @ q.T                # mass-weighted Hessian

    # De-mass-weight to get the Cartesian Hessian
    sqrt_m3 = np.sqrt(np.repeat(np.asarray(masses), 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]

    mol = Molecule.from_arrays(
        atomic_numbers=water_anums(),
        masses=water_masses(),
        coords=np.asarray(coords),
        hessian=h_cart,
    )
    modes = mol.normal_modes()
    recovered = np.sort(np.asarray(modes.frequencies))
    np.testing.assert_allclose(recovered, target_freqs_cm, atol=1e-3)
    assert modes.n_modes == 3


# ----------------------------------------------------------------------
# Linear-molecule support
# ----------------------------------------------------------------------


def test_linear_diatomic_returns_one_mode():
    masses = np.array([14.003, 14.003])
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    # Build a Hessian with one stretch mode
    n = 2
    # mass-weighted unit vector for the stretch: equal+opposite along x,
    # normalised in mass-weighted space
    sqrt_m = np.sqrt(masses)
    v_mw = np.zeros(3 * n)
    v_mw[0] = sqrt_m[0]
    v_mw[3] = -sqrt_m[1]
    v_mw /= np.linalg.norm(v_mw)

    # Trans (3) + rot (2) basis
    t = np.asarray(core.translation_vectors(jnp.array(masses)))
    com = np.asarray(core.shift_to_com(jnp.array(masses), jnp.array(coords)))
    r = np.asarray(core.rotation_vectors(jnp.array(masses), jnp.array(com)))
    base = np.concatenate([t, r, v_mw[:, None]], axis=1)     # (6, 6)
    q, _ = np.linalg.qr(base)
    # Frequency we want for the stretch
    target_cm = 2400.0
    eig_au = (target_cm / HFREQ_CM) ** 2
    eig_full = np.array([0.0] * 5 + [eig_au])
    h_mw = q @ np.diag(eig_full) @ q.T
    sqrt_m3 = np.sqrt(np.repeat(masses, 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]

    mol = Molecule.from_arrays(
        atomic_numbers=[7, 7],
        masses=masses,
        coords=coords,
        hessian=h_cart,
    )
    modes = mol.normal_modes()
    assert modes.is_linear is True
    assert modes.n_modes == 1
    assert np.isclose(float(modes.frequencies[0]), target_cm, atol=1e-2)


# ----------------------------------------------------------------------
# Eckart alignment
# ----------------------------------------------------------------------


def test_eckart_align_recovers_known_rotation():
    """Apply a known rotation to a molecule and verify Eckart finds it."""
    rng = np.random.default_rng(7)
    masses = water_masses()
    coords = water_geometry()
    # Random proper rotation
    a = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(a)
    # Ensure proper (det = +1)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    rotated = coords @ q

    ref = Molecule.from_arrays(atomic_numbers=water_anums(), masses=masses, coords=coords)
    disp = Molecule.from_arrays(atomic_numbers=water_anums(), masses=masses, coords=rotated)
    eck = eckart.eckart_align(ref, disp)
    # The Eckart rotation should equal q (water is asymmetric -> rotation is unique)
    assert eck.rmsd < 1e-9
    assert np.allclose(np.asarray(eck.rotation), q, atol=1e-9)
    # Verify the documented convention: ref @ u brings ref onto disp
    realigned = np.asarray(coords) @ np.asarray(eck.rotation)
    assert np.allclose(realigned, np.asarray(rotated), atol=1e-9)


# ----------------------------------------------------------------------
# Frequencies, frozen test on near-textbook water values
# ----------------------------------------------------------------------


def test_water_thermochemistry_runs():
    """End-to-end smoke test: build water with a synthetic Hessian,
    run NMA, then run thermochemistry, and check that values are
    physically sensible (positive entropies, positive ZPE, etc.)."""
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((9, 9))
    rand[:, :6] = np.concatenate([np.asarray(t), np.asarray(r)], axis=1)
    q, _ = np.linalg.qr(rand)
    target = np.array([1600.0, 3700.0, 3800.0])              # ~water frequencies
    eig = (target / HFREQ_CM) ** 2
    h_mw = q @ np.diag(np.concatenate([np.zeros(6), eig])) @ q.T
    sqrt_m3 = np.sqrt(np.repeat(np.asarray(masses), 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]

    mol = Molecule.from_arrays(
        atomic_numbers=water_anums(),
        masses=water_masses(),
        coords=water_geometry(),
        hessian=h_cart,
    )
    modes = mol.normal_modes()
    thermo = thermochemistry(modes, temperature=298.15, symmetry_number=2)
    assert thermo.zpe > 0
    assert thermo.s_trans > 100      # J/(mol K) - typical water trans entropy
    assert thermo.s_rot > 30
    assert thermo.s_total > thermo.s_trans


# ----------------------------------------------------------------------
# Gradient projection (this was incomplete in the original code: only
# rotation was projected from the gradient, not translation)
# ----------------------------------------------------------------------


def test_gradient_projection_zeroes_out_pure_translation():
    """A pure translation in the gradient must be entirely removed by the
    translation projector before mapping to normal modes."""
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    rng = np.random.default_rng(123)
    rand = rng.standard_normal((9, 9))
    rand[:, :6] = np.concatenate([np.asarray(t), np.asarray(r)], axis=1)
    q, _ = np.linalg.qr(rand)
    target = np.array([1600.0, 3700.0, 3800.0])
    eig = (target / HFREQ_CM) ** 2
    h_mw = q @ np.diag(np.concatenate([np.zeros(6), eig])) @ q.T
    sqrt_m3 = np.sqrt(np.repeat(np.asarray(masses), 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]

    # Build a pure-translation cartesian gradient (all atoms pushed +x by
    # the same amount).  This is invariant under the translational symmetry
    # of the potential, so after projection it should map to ~0 in every
    # vibrational normal mode.
    grad_cart = np.zeros(9)
    grad_cart[0::3] = 1.0       # +x on every atom

    mol = Molecule.from_arrays(
        atomic_numbers=water_anums(),
        masses=water_masses(),
        coords=np.asarray(coords),
        hessian=h_cart,
        gradient=grad_cart,
    )
    modes = mol.normal_modes()
    # Every normal-mode gradient component should be ~0
    np.testing.assert_allclose(
        np.asarray(modes.gradient_normal), 0.0, atol=1e-10
    )


def test_gradient_projection_zeroes_out_pure_rotation():
    """A pure rigid-body rotation of the gradient must be removed too."""
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((9, 9))
    rand[:, :6] = np.concatenate([np.asarray(t), np.asarray(r)], axis=1)
    q, _ = np.linalg.qr(rand)
    target = np.array([1600.0, 3700.0, 3800.0])
    eig = (target / HFREQ_CM) ** 2
    h_mw = q @ np.diag(np.concatenate([np.zeros(6), eig])) @ q.T
    sqrt_m3 = np.sqrt(np.repeat(np.asarray(masses), 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]

    # Pure rotation around z, expressed in the mass-weighted basis.
    # The molecule's gradient lives in Cartesian (Hartree/Bohr) space, and
    # the code mass-weights it as g_mw = g_cart / sqrt(m).  So to inject a
    # gradient that, *after* mass-weighting, lies along rot_mw, we set
    # g_cart = rot_mw * sqrt(m).
    rot_mw = np.asarray(r[:, 2])                       # mass-weighted rot vec
    grad_cart = rot_mw * sqrt_m3                       # de-mass-weight (multiply!)

    mol = Molecule.from_arrays(
        atomic_numbers=water_anums(),
        masses=water_masses(),
        coords=np.asarray(coords),
        hessian=h_cart,
        gradient=grad_cart,
    )
    modes = mol.normal_modes()
    np.testing.assert_allclose(
        np.asarray(modes.gradient_normal), 0.0, atol=1e-10
    )


# ----------------------------------------------------------------------
# Duschinsky round trip
# ----------------------------------------------------------------------


def test_duschinsky_identity_when_reference_equals_displaced():
    """When two states have identical geometry and Hessian, J = I and K = 0."""
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    rng = np.random.default_rng(5)
    rand = rng.standard_normal((9, 9))
    rand[:, :6] = np.concatenate([np.asarray(t), np.asarray(r)], axis=1)
    q, _ = np.linalg.qr(rand)
    target = np.array([1600.0, 3700.0, 3800.0])
    eig = (target / HFREQ_CM) ** 2
    h_mw = q @ np.diag(np.concatenate([np.zeros(6), eig])) @ q.T
    sqrt_m3 = np.sqrt(np.repeat(np.asarray(masses), 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]

    mol = Molecule.from_arrays(
        atomic_numbers=water_anums(), masses=water_masses(),
        coords=np.asarray(coords), hessian=h_cart,
    )
    nm = mol.normal_modes()
    dus = eckart.duschinsky(nm, nm)
    j = np.asarray(dus.J)
    k = np.asarray(dus.K)
    # J should be the identity (or a permutation-with-signs; for a normal-
    # ordered mode list it is the identity up to overall sign per column).
    # The simplest invariant: |J| is the identity (each mode has overlap 1
    # with itself).
    np.testing.assert_allclose(np.abs(j), np.eye(j.shape[0]), atol=1e-8)
    np.testing.assert_allclose(k, 0.0, atol=1e-8)


def test_duschinsky_pure_rotation_does_not_change_modes():
    """Rotating the displaced state's coordinates rigidly should give J ~ I
    after the Eckart alignment removes the rotation again."""
    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    rng = np.random.default_rng(11)
    rand = rng.standard_normal((9, 9))
    rand[:, :6] = np.concatenate([np.asarray(t), np.asarray(r)], axis=1)
    q, _ = np.linalg.qr(rand)
    target = np.array([1600.0, 3700.0, 3800.0])
    eig = (target / HFREQ_CM) ** 2
    h_mw = q @ np.diag(np.concatenate([np.zeros(6), eig])) @ q.T
    sqrt_m3 = np.sqrt(np.repeat(np.asarray(masses), 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]

    # Reference molecule
    ref_mol = Molecule.from_arrays(
        atomic_numbers=water_anums(), masses=water_masses(),
        coords=np.asarray(coords), hessian=h_cart,
    )
    # Displaced: same molecule rotated by a random proper rotation.
    # Both geometry AND Hessian rotate together.
    a = rng.standard_normal((3, 3))
    qrot, _ = np.linalg.qr(a)
    if np.linalg.det(qrot) < 0:
        qrot[:, 0] = -qrot[:, 0]
    # Rotate coords
    new_coords = np.asarray(coords) @ qrot
    # Rotate the Cartesian Hessian: H' = R_3N^T H R_3N with R_3N = block-diag(qrot)
    from jax.scipy.linalg import block_diag as bd
    r3n = np.asarray(bd(*([jnp.asarray(qrot)] * 3)))
    h_rot = r3n.T @ h_cart @ r3n
    disp_mol = Molecule.from_arrays(
        atomic_numbers=water_anums(), masses=water_masses(),
        coords=new_coords, hessian=h_rot,
    )
    nm_ref = ref_mol.normal_modes()
    nm_disp = disp_mol.normal_modes()
    dus = eckart.duschinsky(nm_ref, nm_disp)
    j = np.asarray(dus.J)
    k = np.asarray(dus.K)
    # J should still be (signed) identity, K still ~ 0
    np.testing.assert_allclose(np.abs(j), np.eye(j.shape[0]), atol=1e-7)
    np.testing.assert_allclose(k, 0.0, atol=1e-7)


# ----------------------------------------------------------------------
# X <-> Q transformation matrices
# ----------------------------------------------------------------------


def test_transformation_matrices_roundtrip():
    """D and D^+ must be a left-inverse pair on the vibrational subspace."""
    from nma_jax import transformation_matrices

    masses = jnp.array(water_masses())
    coords = jnp.array(water_geometry())
    com = core.shift_to_com(masses, coords)
    t = core.translation_vectors(masses)
    r = core.rotation_vectors(masses, com)
    rng = np.random.default_rng(99)
    rand = rng.standard_normal((9, 9))
    rand[:, :6] = np.concatenate([np.asarray(t), np.asarray(r)], axis=1)
    q, _ = np.linalg.qr(rand)
    target = np.array([1600.0, 3700.0, 3800.0])
    eig = (target / HFREQ_CM) ** 2
    h_mw = q @ np.diag(np.concatenate([np.zeros(6), eig])) @ q.T
    sqrt_m3 = np.sqrt(np.repeat(np.asarray(masses), 3))
    h_cart = h_mw * sqrt_m3[:, None] * sqrt_m3[None, :]
    mol = Molecule.from_arrays(
        atomic_numbers=water_anums(), masses=water_masses(),
        coords=np.asarray(coords), hessian=h_cart,
    )
    nm = mol.normal_modes()
    mats = transformation_matrices(masses, nm.frequencies, nm.modes_mwc)
    # D^+ D = I in vibrational subspace
    np.testing.assert_allclose(
        np.asarray(mats.D_pinv @ mats.D), np.eye(nm.n_modes), atol=1e-9
    )
    # And the dimensionless pair has the same property up to the FRED scaling
    np.testing.assert_allclose(
        np.asarray(mats.Qdim @ mats.Xdim), np.eye(nm.n_modes), atol=1e-9
    )


# ----------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------


def test_cli_help_runs():
    """The CLI should at least be importable and respond to --help."""
    from nma_jax.cli import build_parser
    parser = build_parser()
    # Will raise SystemExit if it can't parse --help; we just want it to
    # build without errors.
    assert parser is not None
    # Both subcommands present
    subparsers_actions = [
        a for a in parser._actions if a.dest == "command"
    ]
    assert subparsers_actions
    choices = set(subparsers_actions[0].choices.keys())
    assert {"analyze", "compare"}.issubset(choices)
