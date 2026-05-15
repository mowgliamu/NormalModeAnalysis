"""Physical constants and unit conversions.

All values are CODATA 2018 unless otherwise noted. This module is the
single source of truth — never redefine constants elsewhere.
"""
from __future__ import annotations

import math
from typing import Final

# -----------------------------------------------------------------------------
# SI base constants (CODATA 2018)
# -----------------------------------------------------------------------------
PLANCK: Final[float] = 6.626_070_15e-34                # J s
HBAR: Final[float] = PLANCK / (2.0 * math.pi)          # J s
BOLTZMANN: Final[float] = 1.380_649e-23                # J / K
AVOGADRO: Final[float] = 6.022_140_76e23               # 1 / mol
SPEED_OF_LIGHT: Final[float] = 299_792_458.0           # m / s (exact)
GAS_CONSTANT: Final[float] = BOLTZMANN * AVOGADRO      # J / mol / K
ELEMENTARY_CHARGE: Final[float] = 1.602_176_634e-19    # C (exact)
ELECTRON_MASS: Final[float] = 9.109_383_7015e-31       # kg
ATOMIC_MASS_UNIT: Final[float] = 1.660_539_066_60e-27  # kg
BOHR_RADIUS: Final[float] = 5.291_772_109_03e-11       # m
HARTREE_ENERGY: Final[float] = 4.359_744_722_2071e-18  # J

# -----------------------------------------------------------------------------
# Convenient conversions
# -----------------------------------------------------------------------------
HARTREE_TO_EV: Final[float] = 27.211_386_245_988
HARTREE_TO_KJMOL: Final[float] = HARTREE_ENERGY * AVOGADRO / 1000.0  # 2625.4996 kJ/mol
HARTREE_TO_KCALMOL: Final[float] = HARTREE_TO_KJMOL / 4.184          # 627.5095
HARTREE_TO_CM: Final[float] = HARTREE_ENERGY / (PLANCK * SPEED_OF_LIGHT * 100.0)
EV_TO_CM: Final[float] = HARTREE_TO_CM / HARTREE_TO_EV
BOHR_TO_ANG: Final[float] = BOHR_RADIUS * 1.0e10                     # 0.52917721
ANG_TO_BOHR: Final[float] = 1.0 / BOHR_TO_ANG
KCAL_PER_MOL_TO_EV: Final[float] = 1000.0 * 4.184 / (AVOGADRO * ELEMENTARY_CHARGE)

# -----------------------------------------------------------------------------
# Vibrational analysis: angular frequency in atomic units -> wavenumbers (cm^-1)
#
# omega_au = sqrt(eigenvalue of M^{-1/2} H M^{-1/2})  where H is in Hartree/Bohr^2
#            and masses are in electron-mass units. We carry masses in amu, so
#            the mass-weighting needs an extra factor of sqrt(m_e/amu) which
#            absorbs into the conversion factor below.
#
# nu_tilde [cm^-1] = (1 / 2 pi c[cm/s]) * sqrt(E_h / (amu * a0^2))
# -----------------------------------------------------------------------------
HARTREE_OVER_AMU_BOHR2_TO_RADPS: Final[float] = math.sqrt(
    HARTREE_ENERGY / (ATOMIC_MASS_UNIT * BOHR_RADIUS**2)
)
# Numerical value: ~4.5564e16 rad/s for unit eigenvalue
FREQ_AU_TO_CM: Final[float] = HARTREE_OVER_AMU_BOHR2_TO_RADPS / (
    2.0 * math.pi * SPEED_OF_LIGHT * 100.0
)  # ~ 5140.48 cm^-1 -- matches the old constant


# -----------------------------------------------------------------------------
# Dimensionless normal coordinate scaling. The dimensionless coordinate q is
# related to the mass-weighted normal coordinate Q by  q = sqrt(omega / hbar) Q.
# In atomic units (hbar = 1) with omega in atomic units and Q in sqrt(amu)*Bohr,
# we need the prefactor   fred = sqrt(amu * a0^2 / hbar) * sqrt(omega_au).
# The numerical value of sqrt(amu * a0^2 / hbar) [for omega in rad/s] ... but
# the original code used omega expressed via the wavenumber conversion factor.
# We keep both formulations available.
# -----------------------------------------------------------------------------
DIMLESS_PREFACTOR: Final[float] = 0.091_135_5  # historical "fred"; sqrt(amu*a0^2*E_h)/hbar


__all__ = [
    "PLANCK",
    "HBAR",
    "BOLTZMANN",
    "AVOGADRO",
    "SPEED_OF_LIGHT",
    "GAS_CONSTANT",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS",
    "ATOMIC_MASS_UNIT",
    "BOHR_RADIUS",
    "HARTREE_ENERGY",
    "HARTREE_TO_EV",
    "HARTREE_TO_KJMOL",
    "HARTREE_TO_KCALMOL",
    "HARTREE_TO_CM",
    "EV_TO_CM",
    "BOHR_TO_ANG",
    "ANG_TO_BOHR",
    "KCAL_PER_MOL_TO_EV",
    "FREQ_AU_TO_CM",
    "DIMLESS_PREFACTOR",
]
