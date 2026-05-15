"""Rigid-rotor / harmonic-oscillator (RRHO) thermochemistry.

This module is new in the modernised package - the original PhD code declared
``zpe`` as an attribute but never computed thermal corrections.  We provide
ZPE, internal-energy and enthalpy corrections, entropy, and Gibbs free energy
at arbitrary :math:`(T, P)`.

References
----------
Standard statistical-mechanics treatment of the partition function, e.g.
McQuarrie, *Statistical Mechanics*, §6.

The Gaussian whitepaper (``thermochemistry.pdf``) gives the exact formulae
this code reproduces.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import Array

from .constants import (
    AMU_KG,
    AVOGADRO,
    BOHR_M,
    BOLTZMANN,
    GAS_CONSTANT,
    HARTREE_J,
    HARTREE_KELVIN,
    HBAR,
    PLANCK_H,
    SPEED_OF_LIGHT,
)
from .molecule import NormalModes

# Standard-state pressure (1 atm)
P_STANDARD: float = 101_325.0  # Pa


@dataclass(frozen=True, slots=True)
class Thermo:
    """RRHO thermochemistry result at a given temperature and pressure.

    All energies are in **Hartree** unless ``in_kj_mol`` is used; entropy is
    in J/(mol K); free energies follow the same convention.

    Attributes
    ----------
    temperature : K
    pressure : Pa
    zpe : Hartree
        Zero-point vibrational energy (sum 1/2 hν over real modes).
    e_thermal : Hartree
        Thermal correction to internal energy, ``E_trans + E_rot + E_vib_thermal``.
        Does **not** include ZPE.
    h_thermal : Hartree
        Enthalpy correction = ``e_thermal + k_B T`` (per molecule, in Hartree).
    s_trans, s_rot, s_vib, s_elec : J/(mol K)
        Component entropies.
    s_total : J/(mol K)
    g_correction : Hartree
        Thermal correction to Gibbs free energy = ``zpe + h_thermal - T*s_total``.
    """
    temperature: float
    pressure: float
    zpe: float
    e_thermal: float
    h_thermal: float
    s_trans: float
    s_rot: float
    s_vib: float
    s_elec: float
    s_total: float
    g_correction: float

    def summary(self) -> str:
        return (
            f"RRHO thermochemistry at T = {self.temperature:.2f} K, "
            f"P = {self.pressure:.2f} Pa\n"
            f"  ZPE                          = {self.zpe: .8f}  Hartree\n"
            f"  Thermal correction to U      = {self.e_thermal: .8f}  Hartree\n"
            f"  Thermal correction to H      = {self.h_thermal: .8f}  Hartree\n"
            f"  Thermal correction to G      = {self.g_correction: .8f}  Hartree\n"
            f"  S(trans) = {self.s_trans:7.3f}  S(rot) = {self.s_rot:7.3f}  "
            f"S(vib) = {self.s_vib:7.3f}  S(elec) = {self.s_elec:7.3f}  "
            f"[J / (mol K)]\n"
            f"  S(total) = {self.s_total:7.3f}  J / (mol K)\n"
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _wavenumber_to_temperature(nu_cm: np.ndarray) -> np.ndarray:
    """Vibrational temperature :math:`\\theta_v = h c \\tilde\\nu / k_B` in K."""
    # nu_cm in cm^-1; multiply by c (cm/s) to get s^-1, by h to get J, by 1/k_B to get K
    return PLANCK_H * SPEED_OF_LIGHT * 1e2 * nu_cm / BOLTZMANN


# ----------------------------------------------------------------------
# Partition-function components (entropies, energies)
# ----------------------------------------------------------------------


def translational(total_mass_amu: float, t: float, p: float) -> tuple[float, float]:
    """Return ``(E_trans, S_trans)`` for an ideal gas of structureless particles.

    ``E_trans = (3/2) k_B T`` (per molecule); ``S_trans`` from the
    Sackur--Tetrode equation at pressure ``p``.
    Units: ``E_trans`` in J, ``S_trans`` in J/(mol K).
    """
    m = total_mass_amu * AMU_KG
    e_trans = 1.5 * BOLTZMANN * t

    # Sackur-Tetrode: S/k_B = 5/2 + ln[ (2 pi m kT / h^2)^{3/2} * kT/p ]
    q_factor = (2.0 * np.pi * m * BOLTZMANN * t / PLANCK_H ** 2) ** 1.5
    s_per_molecule = BOLTZMANN * (2.5 + np.log(q_factor * BOLTZMANN * t / p))
    s_mol = s_per_molecule * AVOGADRO            # J/(mol K)
    return e_trans, s_mol


def rotational(
    moi_amu_bohr2: np.ndarray,
    t: float,
    symmetry_number: int,
    linear: bool,
) -> tuple[float, float]:
    """Return ``(E_rot, S_rot)`` for a rigid rotor in the high-T limit.

    Parameters
    ----------
    moi_amu_bohr2 : (3,) array of principal moments of inertia in amu*Bohr^2.
        For a linear molecule, only the two non-zero values matter.
    t : K
    symmetry_number : rotational symmetry number (σ)
    linear : True for linear molecules

    Returns
    -------
    E_rot : J (per molecule)
    S_rot : J/(mol K)
    """
    # Convert MOI to SI
    moi_si = np.asarray(moi_amu_bohr2) * AMU_KG * BOHR_M ** 2

    if linear:
        # Use the non-zero principal moment (largest of the two equal MOIs)
        i_b = float(np.max(moi_si))
        theta_b = HBAR ** 2 / (2.0 * i_b * BOLTZMANN)            # rotational temperature
        q_rot = t / (symmetry_number * theta_b)
        e_rot = BOLTZMANN * t                                     # 2/2 kT (two rot DOF)
        s_per = BOLTZMANN * (np.log(q_rot) + 1.0)
    else:
        # Three non-zero principal moments
        theta = HBAR ** 2 / (2.0 * moi_si * BOLTZMANN)             # (3,)
        q_rot = np.sqrt(np.pi) / symmetry_number * np.sqrt(t ** 3 / theta.prod())
        e_rot = 1.5 * BOLTZMANN * t
        s_per = BOLTZMANN * (np.log(q_rot) + 1.5)

    s_mol = s_per * AVOGADRO
    return e_rot, s_mol


def vibrational(
    frequencies_cm: np.ndarray,
    t: float,
    *,
    imaginary_cutoff: float = 0.0,
) -> tuple[float, float, float]:
    """Return ``(ZPE, E_vib_thermal, S_vib)`` for harmonic oscillators.

    Imaginary frequencies (``ν < imaginary_cutoff``) are excluded - they
    correspond to transition-state modes or numerical noise, and a true
    transition-state Gibbs free energy should be computed with the
    imaginary mode dropped.

    Returns
    -------
    zpe : J (per molecule)
    e_vib_thermal : J (per molecule); does **not** include ZPE
    s_vib : J/(mol K)
    """
    nu = np.asarray(frequencies_cm)
    real = nu[nu > imaginary_cutoff]
    if real.size == 0:
        return 0.0, 0.0, 0.0
    theta_v = _wavenumber_to_temperature(real)                 # K, per mode
    # ZPE = sum (1/2) hν, in J per molecule
    nu_hz = real * SPEED_OF_LIGHT * 1e2                         # cm^-1 -> Hz
    zpe = 0.5 * PLANCK_H * nu_hz.sum()
    # Use x = theta_v / T
    x = theta_v / t
    exp_x = np.expm1(x)                                         # exp(x) - 1, stable
    # E_thermal per mode = kT * x / (exp(x) - 1)
    e_thermal = (BOLTZMANN * t * (x / exp_x)).sum()
    # S per molecule per mode = k * [ x/(exp(x) - 1) - ln(1 - exp(-x)) ]
    s_per = BOLTZMANN * ((x / exp_x) - np.log1p(-np.exp(-x))).sum()
    return float(zpe), float(e_thermal), float(s_per * AVOGADRO)


def electronic(spin_multiplicity: int, t: float) -> tuple[float, float]:
    """Electronic ``(E_elec, S_elec)`` assuming ground-state-only population.

    Returns ``(0.0, R ln g_0)`` where ``g_0 = 2S + 1``.  Excited electronic
    states are not included; for systems with low-lying excited states the
    caller should add their contribution manually.
    """
    g0 = spin_multiplicity
    s = GAS_CONSTANT * np.log(g0)
    return 0.0, s


# ----------------------------------------------------------------------
# Top-level convenience
# ----------------------------------------------------------------------


def thermochemistry(
    modes: NormalModes,
    *,
    temperature: float = 298.15,
    pressure: float = P_STANDARD,
    symmetry_number: int = 1,
    spin_multiplicity: int = 1,
    imaginary_cutoff: float = 0.0,
) -> Thermo:
    """Compute RRHO thermochemistry at ``(T, P)``.

    Parameters
    ----------
    modes : NormalModes
        Output of :meth:`Molecule.normal_modes`.
    temperature : K, default 298.15
    pressure : Pa, default 101325 (1 atm)
    symmetry_number : int, default 1
        Rotational symmetry number σ.  Set this correctly for your molecule:
        e.g. 2 for water, 3 for ammonia, 12 for methane.
    spin_multiplicity : int, default 1
        2S+1 for the ground electronic state.
    imaginary_cutoff : float, default 0
        Frequencies below this (cm^-1) are excluded from the vibrational
        sum.  For transition states pass ``0.0`` (drops the single imaginary
        mode automatically).
    """
    mol = modes.molecule
    total_mass = float(mol.masses.sum())
    moi = np.asarray(mol.principal_moments)
    linear = bool(modes.is_linear)
    freqs = np.asarray(modes.frequencies)

    e_trans, s_trans = translational(total_mass, temperature, pressure)
    e_rot, s_rot = rotational(moi, temperature, symmetry_number, linear)
    zpe_j, e_vib, s_vib = vibrational(freqs, temperature, imaginary_cutoff=imaginary_cutoff)
    _, s_elec = electronic(spin_multiplicity, temperature)

    # Total internal-energy thermal correction (NOT including ZPE)
    e_total = e_trans + e_rot + e_vib
    h_total = e_total + BOLTZMANN * temperature       # for an ideal gas
    s_total = s_trans + s_rot + s_vib + s_elec

    # Convert per-molecule energies to Hartree (multiply by 1/HARTREE_J)
    zpe_hartree = zpe_j / HARTREE_J
    e_thermal_hartree = e_total / HARTREE_J
    h_thermal_hartree = h_total / HARTREE_J
    # Free-energy correction in Hartree: ZPE + H_thermal - TS_total
    # T*S_total per mole, convert to Hartree per molecule
    ts_per_molecule = temperature * s_total / AVOGADRO             # J per molecule
    g_correction = zpe_hartree + h_thermal_hartree - ts_per_molecule / HARTREE_J

    return Thermo(
        temperature=temperature,
        pressure=pressure,
        zpe=zpe_hartree,
        e_thermal=e_thermal_hartree,
        h_thermal=h_thermal_hartree,
        s_trans=s_trans,
        s_rot=s_rot,
        s_vib=s_vib,
        s_elec=s_elec,
        s_total=s_total,
        g_correction=g_correction,
    )
