"""Physical constants and unit conversions.

All values use CODATA 2018 recommended values. This module is the single
source of truth - do not redefine these elsewhere in the package.

References
----------
P. J. Mohr, B. N. Taylor, D. B. Newell, CODATA Recommended Values of
the Fundamental Physical Constants: 2018.
"""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Fundamental SI constants (CODATA 2018)
# ----------------------------------------------------------------------
PLANCK_H: Final[float] = 6.62607015e-34          # J s          (exact, SI 2019)
HBAR: Final[float] = 1.054571817e-34             # J s
SPEED_OF_LIGHT: Final[float] = 299_792_458.0     # m s^-1       (exact)
BOLTZMANN: Final[float] = 1.380649e-23           # J K^-1       (exact, SI 2019)
AVOGADRO: Final[float] = 6.02214076e23           # mol^-1       (exact, SI 2019)
ELEM_CHARGE: Final[float] = 1.602176634e-19      # C            (exact, SI 2019)
GAS_CONSTANT: Final[float] = AVOGADRO * BOLTZMANN  # J mol^-1 K^-1
AMU_KG: Final[float] = 1.66053906660e-27         # kg
ELECTRON_MASS_KG: Final[float] = 9.1093837015e-31  # kg

# ----------------------------------------------------------------------
# Atomic units
# ----------------------------------------------------------------------
HARTREE_J: Final[float] = 4.3597447222071e-18    # J
BOHR_M: Final[float] = 5.29177210903e-11         # m

# ----------------------------------------------------------------------
# Commonly used conversion factors
# ----------------------------------------------------------------------
HARTREE_EV: Final[float] = 27.211386245988
HARTREE_KCALMOL: Final[float] = 627.5094740631
HARTREE_KJMOL: Final[float] = 2625.4996394798
HARTREE_CM_INV: Final[float] = 219_474.6313632   # cm^-1 per Hartree
HARTREE_KELVIN: Final[float] = 315_775.02480407  # K per Hartree
EV_CM_INV: Final[float] = 8065.543937
KCALMOL_EV: Final[float] = 0.04336410390
BOHR_ANGSTROM: Final[float] = 0.529_177_210_903
ANGSTROM_BOHR: Final[float] = 1.0 / BOHR_ANGSTROM
AMU_ELECTRON_MASS: Final[float] = AMU_KG / ELECTRON_MASS_KG  # ~1822.888486

# ----------------------------------------------------------------------
# Vibrational-analysis convenience factor
# ----------------------------------------------------------------------
# Converts sqrt(eigenvalue of mass-weighted Hessian) to wavenumbers
# when the Hessian is in atomic units (Hartree / Bohr^2) and masses in amu:
#
#     nu_tilde [cm^-1] = sqrt(lambda) * HFREQ_CM
#
# Derivation:
#     omega [au]  = sqrt(k/m)
#     omega [s^-1] = omega[au] * sqrt(E_h / (m_e * a0^2))
#     nu_tilde [cm^-1] = omega / (2*pi*c) [in cm units]
#     Account for mass in amu vs m_e: factor sqrt(m_e/amu) = 1/sqrt(1822.89)
#
# The numerical value below uses the CODATA 2018 inputs and reproduces
# Gaussian / molpro / Q-Chem to ~6 significant figures.
HFREQ_CM: Final[float] = 5140.484532              # historical value: 5140.48

# Dimensionless-normal-coordinate factor:
#     q_alpha = FRED * sqrt(omega_cm) * Q_mw
# where Q_mw is mass-weighted normal coord in (amu^{1/2} * Bohr).
# Derived from: FRED = sqrt(amu / (m_e * HARTREE_CM_INV))
FRED: Final[float] = 0.091135502

# ----------------------------------------------------------------------
# Atom symbols indexed by atomic number (Z=0 sentinel + Z=1..118)
# ----------------------------------------------------------------------
ATOMIC_SYMBOLS: Final[tuple[str, ...]] = (
    "X",
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
    "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)


def symbol_for(z: int) -> str:
    """Return the element symbol for atomic number ``z`` (1-based)."""
    if z < 0 or z >= len(ATOMIC_SYMBOLS):
        raise ValueError(f"Atomic number {z} out of range 0..{len(ATOMIC_SYMBOLS) - 1}")
    return ATOMIC_SYMBOLS[z]
