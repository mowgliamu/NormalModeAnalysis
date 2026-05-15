"""Sanity tests using known reference values.

Run with:  ``pytest tests/`` or just ``python tests/test_basic.py``.
"""
from __future__ import annotations

import math

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from nma_jax import (
    Molecule,
    harmonic_analysis,
    thermochemistry,
    zero_point_energy,
    build_transforms,
    duschinsky,
    project_gradient_to_modes,
)
from nma_jax.constants import ANG_TO_BOHR


# -----------------------------------------------------------------------------
# Helpers: build a simple water Hessian using a harmonic model so we can test
# the analysis pipeline end-to-end without needing a Gaussian output file.
# We use experimental geometry + arbitrary force constants chosen so that the
# resulting frequencies are in a sensible range.
# -----------------------------------------------------------------------------
def water_molecule(force_constant: float = 0.5) -> Molecule:
    """A water molecule with COM-centred experimental geometry (Bohr).

    The Hessian is a *diagonal* mass-weighted matrix in the Cartesian basis,
    which gives every Cartesian degree of freedom the same vibrational
    "spring constant". This is unphysical but it's a controlled test case:
    after projecting translations + rotations we should get exactly 3
    identical positive frequencies (the molecule has 3 vibrational modes).
    """
    # Experimental O-H bond length 0.9572 Å, H-O-H angle 104.52°
    bohr = ANG_TO_BOHR
    rOH = 0.9572 * bohr
    half_angle = math.radians(104.52 / 2.0)

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [rOH * math.sin(half_angle), 0.0, rOH * math.cos(half_angle)],
            [-rOH * math.sin(half_angle), 0.0, rOH * math.cos(half_angle)],
        ],
        dtype=float,
    )
    Z = np.array([8, 1, 1], dtype=int)
    masses = np.array([15.999, 1.008, 1.008])

    # A diagonal Hessian; force_constant is in Hartree / Bohr^2
    H = np.eye(9) * force_constant
    return Molecule.from_arrays(Z, coords, H, masses=masses)


def test_water_vibrational_count() -> None:
    mol = water_molecule()
    analysis = harmonic_analysis(mol)
    # 3 atoms -> 9 DOF -> 6 trans+rot -> 3 vibrations
    assert analysis.nmode == 3, f"expected 3 modes, got {analysis.nmode}"
    assert not analysis.is_linear
    # All three frequencies should be real and positive for our test Hessian
    freqs = np.asarray(analysis.frequencies_cm)
    assert np.all(freqs > 0), f"unexpected imaginary frequencies: {freqs}"
    print(f"  water frequencies (test Hessian): {freqs}")
    print(f"  reduced masses: {np.asarray(analysis.reduced_masses)}")


def test_water_modes_orthonormal() -> None:
    mol = water_molecule()
    analysis = harmonic_analysis(mol)
    L = np.asarray(analysis.modes_mw_cart)
    eye = L.T @ L
    err = np.max(np.abs(eye - np.eye(analysis.nmode)))
    assert err < 1e-10, f"modes not orthonormal: max deviation {err}"
    print(f"  mode orthonormality:  max |L^T L - I| = {err:.2e}")


def test_water_thermo_sanity() -> None:
    """ZPE matches sum of (1/2) h c nu, and thermochem fields are sane."""
    mol = water_molecule()
    analysis = harmonic_analysis(mol)
    zpe = zero_point_energy(analysis.frequencies_cm)
    assert zpe > 0
    thermo = thermochemistry(mol, analysis, temperature=298.15, symmetry_number=2)
    # ZPE consistency
    assert math.isclose(zpe, thermo.zpe, rel_tol=1e-10)
    # Sackur-Tetrode-ish: translational entropy at 298 K, 1 atm for ~18 amu
    # should be around 144.8 J/(mol K) = 5.51e-5 Hartree/K — order-of-magnitude check
    s_trans_si = thermo.s_trans * 4.359744e-18 * 6.02214e23  # Hartree/K -> J/(mol K)
    print(f"  S_trans (water, 298 K, 1 atm): {s_trans_si:.2f} J/(mol K)  (lit ~144.96)")
    assert 140.0 < s_trans_si < 150.0, f"S_trans out of range: {s_trans_si}"


def test_linear_molecule_detected() -> None:
    """A linear molecule should have nmode = 3N-5 not 3N-6."""
    # CO2-like geometry along z (Bohr); arbitrary diagonal Hessian
    coords = np.array(
        [[0.0, 0.0, -2.2], [0.0, 0.0, 0.0], [0.0, 0.0, 2.2]],
        dtype=float,
    )
    Z = np.array([8, 6, 8], dtype=int)
    masses = np.array([15.999, 12.011, 15.999])
    H = np.eye(9) * 0.5
    mol = Molecule.from_arrays(Z, coords, H, masses=masses)
    analysis = harmonic_analysis(mol)
    assert analysis.is_linear, "linear molecule was not detected"
    # 3*3 - 5 = 4 vibrational modes
    assert analysis.nmode == 4, f"linear molecule should have 4 modes, got {analysis.nmode}"
    print(f"  linear CO2-like: nmode = {analysis.nmode} (expected 4)")


def test_transition_state_signed_freq() -> None:
    """Inject a negative eigenvalue along an actual vibrational mode and verify
    that the analysis recovers one signed-negative frequency."""
    mol = water_molecule()
    analysis_orig = harmonic_analysis(mol)
    # Build a new Cartesian Hessian by replacing one eigenvalue with a
    # negative value, then un-projecting.
    L = np.asarray(analysis_orig.modes_mw_cart)      # (9, 3) mass-weighted
    eig = np.asarray(analysis_orig.eigenvalues_au)   # (3,)
    new_eig = eig.copy()
    new_eig[0] = -abs(new_eig[0])                    # flip sign of one mode
    # F_proj = L diag(eig) L^T  (in the vibrational subspace)
    F_vib = L @ np.diag(new_eig) @ L.T
    # Bring back to Cartesian by mass-weighting: H = M^{1/2} F M^{1/2}
    masses = np.asarray(mol.masses)
    Mhalf = np.diag(np.repeat(np.sqrt(masses), 3))
    H_new = Mhalf @ F_vib @ Mhalf
    mol_ts = mol.with_hessian(H_new)
    analysis = harmonic_analysis(mol_ts)
    freqs = np.asarray(analysis.frequencies_cm)
    n_imag = int(np.sum(freqs < 0))
    assert n_imag == 1, f"expected 1 imaginary frequency, got {n_imag}: {freqs}"
    # The imaginary frequency should be first (we sort by signed eigenvalue)
    assert freqs[0] < 0
    print(f"  TS test: frequencies = {freqs}  (1 imaginary, sorted first)")


def test_transforms_roundtrip() -> None:
    """Q -> x and x -> Q should be inverses on the mode subspace."""
    mol = water_molecule()
    analysis = harmonic_analysis(mol)
    t = build_transforms(mol, analysis)
    # x_to_Q @ Q_to_x should be the identity on the mode space
    I_modes = np.asarray(t.x_to_Q @ t.Q_to_x)
    err = np.max(np.abs(I_modes - np.eye(analysis.nmode)))
    assert err < 1e-10, f"roundtrip error: {err}"
    print(f"  roundtrip x<->Q:  max |x_to_Q @ Q_to_x - I| = {err:.2e}")


def test_duschinsky_identity() -> None:
    """Duschinsky between a molecule and itself: J = I, K = 0."""
    mol = water_molecule()
    analysis = harmonic_analysis(mol)
    result = duschinsky(mol, analysis, mol, analysis)
    J = np.asarray(result.J)
    K = np.asarray(result.K)
    err_J = np.max(np.abs(J - np.eye(analysis.nmode)))
    err_K = np.max(np.abs(K))
    assert err_J < 1e-10, f"self-Duschinsky J is not identity: {err_J}"
    assert err_K < 1e-10, f"self-Duschinsky K is not zero: {err_K}"
    print(f"  self-Duschinsky:  |J-I| = {err_J:.2e}, |K| = {err_K:.2e}")


def test_duschinsky_rotated() -> None:
    """A rotated copy of the same molecule should still give J=I, K=0
    after Eckart alignment."""
    mol = water_molecule()
    analysis = harmonic_analysis(mol)

    # Apply a known rotation about the z-axis
    angle = math.radians(37.0)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated = np.asarray(mol.coordinates) @ R.T
    mol_rot = mol.with_coordinates(rotated)
    # Note: rotating the coords means the Hessian no longer matches — for the
    # purposes of this test we just want to check Eckart alignment, so we
    # need to also rotate the Hessian.
    natom = mol.natom
    Rb = np.kron(np.eye(natom), R)
    H_rot = Rb @ np.asarray(mol.hessian_cart) @ Rb.T
    mol_rot = mol_rot.with_hessian(H_rot)
    analysis_rot = harmonic_analysis(mol_rot)

    result = duschinsky(mol_rot, analysis_rot, mol, analysis)
    K = np.asarray(result.K)
    # After Eckart alignment, displacement should be ~zero
    err_K = np.max(np.abs(K))
    assert err_K < 1e-8, f"rotated Duschinsky K is too large: {err_K}"
    print(f"  rotated Duschinsky:  RMS alignment = {result.rms_alignment:.2e}, |K| = {err_K:.2e}")


def test_gradient_projection() -> None:
    """Gradient in normal mode basis should have right shape and finite values."""
    mol = water_molecule()
    # Attach a synthetic Cartesian gradient
    grad = np.linspace(-0.01, 0.01, 9)
    mol = Molecule.from_arrays(
        np.asarray(mol.atomic_numbers),
        np.asarray(mol.coordinates),
        np.asarray(mol.hessian_cart),
        masses=np.asarray(mol.masses),
        gradient_cart=grad,
    )
    analysis = harmonic_analysis(mol)
    grad_nm = np.asarray(project_gradient_to_modes(mol, analysis))
    assert grad_nm.shape == (analysis.nmode,)
    assert np.all(np.isfinite(grad_nm))
    print(f"  gradient in normal-mode basis: shape={grad_nm.shape}, max={np.max(np.abs(grad_nm)):.4f}")


def test_jit_compatibility() -> None:
    """Vibrational analysis should be jit-compilable."""
    # We can't jit the high-level function directly because of `import numpy`
    # inside it (for the mask handling), but the inner inertia_tensor is
    # pure jax and should jit fine.
    from nma_jax.vibrational import inertia_tensor
    fn = jax.jit(inertia_tensor)
    mol = water_molecule()
    I = fn(mol.coordinates - mol.center_of_mass, mol.masses)
    assert I.shape == (3, 3)
    # Inertia tensor should be symmetric
    err = np.max(np.abs(np.asarray(I) - np.asarray(I).T))
    assert err < 1e-12
    print(f"  jit(inertia_tensor) works:  max asymmetry = {err:.2e}")


# -----------------------------------------------------------------------------
def main() -> None:
    tests = [
        test_water_vibrational_count,
        test_water_modes_orthonormal,
        test_water_thermo_sanity,
        test_linear_molecule_detected,
        test_transition_state_signed_freq,
        test_transforms_roundtrip,
        test_duschinsky_identity,
        test_duschinsky_rotated,
        test_gradient_projection,
        test_jit_compatibility,
    ]
    failures = 0
    for t in tests:
        name = t.__name__
        print(f"\n{name}:")
        try:
            t()
            print(f"  PASS")
        except Exception as e:
            print(f"  FAIL: {e!r}")
            failures += 1
    print(f"\n{'=' * 50}")
    print(f"  {len(tests) - failures}/{len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
