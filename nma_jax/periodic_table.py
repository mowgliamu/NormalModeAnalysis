"""Periodic table: atomic numbers, symbols, and standard atomic weights.

Standard atomic weights from IUPAC 2021 (for natural isotopic abundances).
Use these as a fallback when the upstream program does not report masses.
"""
from __future__ import annotations

from typing import Final

ELEMENT_SYMBOLS: Final[tuple[str, ...]] = (
    "X",   # 0 — placeholder so element_symbols[Z] indexing works
    "H",   "He",
    "Li",  "Be",  "B",   "C",   "N",   "O",   "F",   "Ne",
    "Na",  "Mg",  "Al",  "Si",  "P",   "S",   "Cl",  "Ar",
    "K",   "Ca",
    "Sc",  "Ti",  "V",   "Cr",  "Mn",  "Fe",  "Co",  "Ni",  "Cu",  "Zn",
    "Ga",  "Ge",  "As",  "Se",  "Br",  "Kr",
    "Rb",  "Sr",
    "Y",   "Zr",  "Nb",  "Mo",  "Tc",  "Ru",  "Rh",  "Pd",  "Ag",  "Cd",
    "In",  "Sn",  "Sb",  "Te",  "I",   "Xe",
    "Cs",  "Ba",
    "La",  "Ce",  "Pr",  "Nd",  "Pm",  "Sm",  "Eu",  "Gd",  "Tb",  "Dy",
    "Ho",  "Er",  "Tm",  "Yb",  "Lu",
    "Hf",  "Ta",  "W",   "Re",  "Os",  "Ir",  "Pt",  "Au",  "Hg",
    "Tl",  "Pb",  "Bi",  "Po",  "At",  "Rn",
)

# IUPAC 2021 standard atomic weights (amu). NaN where no stable isotope.
# Source: J. Meija et al., Pure Appl. Chem. 88, 265 (2016) and 2021 update.
STANDARD_ATOMIC_WEIGHTS: Final[tuple[float, ...]] = (
    0.0,
    1.008, 4.002602,
    6.94, 9.0121831, 10.81, 12.011, 14.007, 15.999, 18.998403163, 20.1797,
    22.98976928, 24.305, 26.9815385, 28.085, 30.973761998, 32.06, 35.45, 39.948,
    39.0983, 40.078,
    44.955908, 47.867, 50.9415, 51.9961, 54.938044, 55.845, 58.933194, 58.6934,
    63.546, 65.38,
    69.723, 72.630, 74.921595, 78.971, 79.904, 83.798,
    85.4678, 87.62,
    88.90584, 91.224, 92.90637, 95.95, 98.0, 101.07, 102.90550, 106.42,
    107.8682, 112.414,
    114.818, 118.710, 121.760, 127.60, 126.90447, 131.293,
    132.90545196, 137.327,
    138.90547, 140.116, 140.90766, 144.242, 145.0, 150.36, 151.964, 157.25,
    158.92535, 162.500, 164.93033, 167.259, 168.93422, 173.045, 174.9668,
    178.49, 180.94788, 183.84, 186.207, 190.23, 192.217, 195.084, 196.966569,
    200.592,
    204.38, 207.2, 208.98040, 209.0, 210.0, 222.0,
)


def symbol(z: int) -> str:
    """Return element symbol for atomic number ``z``."""
    if not 1 <= z < len(ELEMENT_SYMBOLS):
        raise ValueError(f"Unsupported atomic number: {z}")
    return ELEMENT_SYMBOLS[z]


def standard_weight(z: int) -> float:
    """Return IUPAC standard atomic weight (amu) for atomic number ``z``."""
    if not 1 <= z < len(STANDARD_ATOMIC_WEIGHTS):
        raise ValueError(f"Unsupported atomic number: {z}")
    return STANDARD_ATOMIC_WEIGHTS[z]


__all__ = ["ELEMENT_SYMBOLS", "STANDARD_ATOMIC_WEIGHTS", "symbol", "standard_weight"]
