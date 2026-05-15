"""Rigid-rotor / harmonic-oscillator (RRHO) thermochemistry.

Given a vibrational analysis and a molecule, compute the standard
thermodynamic functions at a chosen temperature and pressure:

- Zero-point energy
- Translational, rotational, vibrational and electronic contributions to
  the internal energy U, enthalpy H, entropy S, Helmholtz energy A,
  Gibbs energy G, and heat capacities Cv / Cp.

Imaginary frequencies are skipped automatically (they contribute nothing
to the partition function; you typically want to compute thermochemistry
for minima, but we don't raise on a transition state — many users want a
"do it anyway" estimate at the TS).

All outputs are in **Hartree** for energies and **Hartree / K** for
entropies and heat capacities, matching Gaussian's printed numbers. Use
``HARTREE_TO_KJMOL`` etc. from :mod:`nma_jax.constants` if you want to
convert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .constants import (
    AVOGADRO,
    BOHR_RADIUS,
    BOLTZMANN,
    ATOMIC_MASS_UNIT,
    HARTREE_ENERGY,
    PLANCK,
    SPEED_OF_LIGHT,
)
from .molecule import Molecule
from .vibrational import VibrationalAnalysis, inertia_tensor

Array = jax.Array

# Standard-state pressure (1 atm), Pa
STANDARD_PRESSURE = 101_325.0


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Thermochemistry:
    """RRHO thermochemistry at a given temperature and pressure.

    All energies in Hartree, entropies and heat capacities in Hartree/K.
    Add ``Molecule.energy`` separately to get total thermodynamic
    functions (E0 + ZPE, E0 + H, G = E0 + H - T S etc.).
    """
    temperature: float
    pressure: float
    symmetry_number: int

    zpe: float
    thermal_correction_to_energy: float    # U(T) - U(0)  (excludes ZPE)
    thermal_correction_to_enthalpy: float  # H(T) - U(0)  (excludes ZPE)
    thermal_correction_to_gibbs: float     # G(T) - U(0)  (excludes ZPE)

    u_trans: float; u_rot: float; u_vib: float; u_elec: float
    s_trans: float; s_rot: float; s_vib: float; s_elec: float
    cv_trans: float; cv_rot: float; cv_vib: float
    cv_total: float; cp_total: float
    s_total: float
    u_total: float
    h_total: float
    g_total: float


# -----------------------------------------------------------------------------
# Building blocks (in SI; we convert at the end)
# -----------------------------------------------------------------------------
def _principal_moments_si(molecule: Molecule) -> np.ndarray:
    """Principal moments of inertia in SI (kg m^2), sorted ascending."""
    coords = np.asarray(molecule.coordinates) * BOHR_RADIUS  # Bohr -> m
    masses = np.asarray(molecule.masses) * ATOMIC_MASS_UNIT  # amu  -> kg
    coords_c = coords - (masses[:, None] * coords).sum(axis=0) / masses.sum()
    I = np.asarray(inertia_tensor(jnp.asarray(coords_c), jnp.asarray(masses)))
    return np.sort(np.linalg.eigvalsh(I))


def zero_point_energy(
    frequencies_cm: Array | np.ndarray, *, scale: float = 1.0
) -> float:
    """ZPE = (1/2) * sum_k h c nu_k  (skipping imaginary modes).

    Parameters
    ----------
    frequencies_cm : array
        Harmonic frequencies in cm^-1 (imaginary modes carry a negative sign).
    scale : float, optional
        Empirical scaling factor (default 1.0).

    Returns
    -------
    float
        ZPE in **Hartree**.
    """
    nu = np.asarray(frequencies_cm)
    real = nu[nu > 0] * scale
    e_joules = 0.5 * np.sum(PLANCK * SPEED_OF_LIGHT * 100.0 * real)  # SI: 1/cm -> 1/m
    return float(e_joules / HARTREE_ENERGY)


def thermochemistry(
    molecule: Molecule,
    analysis: VibrationalAnalysis,
    *,
    temperature: float = 298.15,
    pressure: float = STANDARD_PRESSURE,
    symmetry_number: int = 1,
    scale: float = 1.0,
    electronic_multiplicity: int = 1,
) -> Thermochemistry:
    """Full RRHO thermochemistry. See :class:`Thermochemistry` for fields.

    Notes
    -----
    The symmetry number must be set by the caller (we don't infer point
    groups). 1 for asymmetric molecules, 2 for water, 12 for methane, etc.

    Imaginary frequencies are silently skipped.
    """
    T = float(temperature)
    P = float(pressure)
    if T <= 0:
        raise ValueError("Temperature must be positive.")

    # --- Translation ---------------------------------------------------
    M_kg = float(molecule.total_mass) * ATOMIC_MASS_UNIT
    # Thermal de Broglie wavelength
    Lambda = PLANCK / np.sqrt(2.0 * np.pi * M_kg * BOLTZMANN * T)
    V = BOLTZMANN * T / P                  # molar volume per particle (m^3)
    q_trans = V / Lambda**3
    # Sackur-Tetrode
    s_trans_jk = BOLTZMANN * (np.log(q_trans) + 5.0 / 2.0)
    u_trans_j = 1.5 * BOLTZMANN * T
    cv_trans_jk = 1.5 * BOLTZMANN

    # --- Rotation ------------------------------------------------------
    natom = molecule.natom
    if natom == 1:
        s_rot_jk = u_rot_j = cv_rot_jk = 0.0
    else:
        I_principal = _principal_moments_si(molecule)
        is_linear = analysis.is_linear or (I_principal[0] < 1e-46 * max(I_principal[-1], 1.0))
        if is_linear:
            # Use the largest moment (the two non-zero are equal for linear)
            I_eff = I_principal[-1]
            q_rot = (8.0 * np.pi**2 * I_eff * BOLTZMANN * T) / (
                symmetry_number * PLANCK**2
            )
            u_rot_j = BOLTZMANN * T
            s_rot_jk = BOLTZMANN * (np.log(q_rot) + 1.0)
            cv_rot_jk = BOLTZMANN
        else:
            Ia, Ib, Ic = I_principal
            theta_a = PLANCK**2 / (8.0 * np.pi**2 * Ia * BOLTZMANN)
            theta_b = PLANCK**2 / (8.0 * np.pi**2 * Ib * BOLTZMANN)
            theta_c = PLANCK**2 / (8.0 * np.pi**2 * Ic * BOLTZMANN)
            q_rot = (np.sqrt(np.pi) / symmetry_number) * np.sqrt(
                T**3 / (theta_a * theta_b * theta_c)
            )
            u_rot_j = 1.5 * BOLTZMANN * T
            s_rot_jk = BOLTZMANN * (np.log(q_rot) + 1.5)
            cv_rot_jk = 1.5 * BOLTZMANN

    # --- Vibration -----------------------------------------------------
    nu_cm = np.asarray(analysis.frequencies_cm)
    real = nu_cm[nu_cm > 0] * scale
    # Vibrational temperatures
    theta = (PLANCK * SPEED_OF_LIGHT * 100.0 * real) / BOLTZMANN  # K
    x = theta / T
    # Avoid overflow in exp for huge x: use a safe form for U
    ex = np.exp(np.minimum(x, 700.0))
    u_vib_j = float(np.sum(BOLTZMANN * theta * (0.5 + 1.0 / (ex - 1.0))))
    s_vib_jk = float(
        np.sum(BOLTZMANN * (x / (ex - 1.0) - np.log1p(-1.0 / ex)))
    )
    cv_vib_jk = float(
        np.sum(BOLTZMANN * (x**2 * ex) / (ex - 1.0) ** 2)
    )
    # The vibrational *thermal* energy (above the ZPE) is u_vib_j minus ZPE_J
    zpe_J = 0.5 * np.sum(PLANCK * SPEED_OF_LIGHT * 100.0 * real)
    u_vib_thermal_j = u_vib_j - zpe_J

    # --- Electronic ----------------------------------------------------
    s_elec_jk = BOLTZMANN * np.log(electronic_multiplicity)
    u_elec_j = 0.0

    # --- Aggregates ----------------------------------------------------
    # Thermal corrections (above U(0), which is E_elec + ZPE)
    u_thermal_j = u_trans_j + u_rot_j + u_vib_thermal_j + u_elec_j
    h_thermal_j = u_thermal_j + BOLTZMANN * T  # PV = NkT for ideal gas
    s_total_jk = s_trans_jk + s_rot_jk + s_vib_jk + s_elec_jk
    g_thermal_j = h_thermal_j - T * s_total_jk

    cv_total_jk = cv_trans_jk + cv_rot_jk + cv_vib_jk
    cp_total_jk = cv_total_jk + BOLTZMANN

    # Convert to Hartree (energies) and Hartree/K (entropies, Cv, Cp)
    e_to_au = 1.0 / HARTREE_ENERGY
    s_to_au = 1.0 / HARTREE_ENERGY

    zpe_au = float(zpe_J * e_to_au)
    u_thermal_au = float(u_thermal_j * e_to_au)
    h_thermal_au = float(h_thermal_j * e_to_au)
    g_thermal_au = float(g_thermal_j * e_to_au)

    # Absolute thermodynamic functions referenced to E_elec=0:
    # (caller can add ``molecule.energy`` to recover totals)
    u_total_au = zpe_au + u_thermal_au
    h_total_au = zpe_au + h_thermal_au
    g_total_au = zpe_au + g_thermal_au

    return Thermochemistry(
        temperature=T,
        pressure=P,
        symmetry_number=symmetry_number,
        zpe=zpe_au,
        thermal_correction_to_energy=u_thermal_au,
        thermal_correction_to_enthalpy=h_thermal_au,
        thermal_correction_to_gibbs=g_thermal_au,
        u_trans=float(u_trans_j * e_to_au),
        u_rot=float(u_rot_j * e_to_au),
        u_vib=float(u_vib_thermal_j * e_to_au),
        u_elec=float(u_elec_j * e_to_au),
        s_trans=float(s_trans_jk * s_to_au),
        s_rot=float(s_rot_jk * s_to_au),
        s_vib=float(s_vib_jk * s_to_au),
        s_elec=float(s_elec_jk * s_to_au),
        cv_trans=float(cv_trans_jk * s_to_au),
        cv_rot=float(cv_rot_jk * s_to_au),
        cv_vib=float(cv_vib_jk * s_to_au),
        cv_total=float(cv_total_jk * s_to_au),
        cp_total=float(cp_total_jk * s_to_au),
        s_total=float(s_total_jk * s_to_au),
        u_total=u_total_au,
        h_total=h_total_au,
        g_total=g_total_au,
    )


__all__ = ["Thermochemistry", "zero_point_energy", "thermochemistry", "STANDARD_PRESSURE"]
