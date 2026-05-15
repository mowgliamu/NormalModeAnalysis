"""Output utilities: pretty-print analyses, write transform tables, and
produce Molden-format files for visualizing normal modes.
"""
from __future__ import annotations

from pathlib import Path
from typing import TextIO

import numpy as np

from .constants import BOHR_TO_ANG
from .molecule import Molecule
from .transformations import NormalCoordinateTransform
from .vibrational import VibrationalAnalysis


# -----------------------------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------------------------
def format_frequencies(analysis: VibrationalAnalysis) -> str:
    """Return a human-readable frequency table.

    Imaginary frequencies appear with a leading minus and (i) flag.
    """
    freqs = np.asarray(analysis.frequencies_cm)
    mu = np.asarray(analysis.reduced_masses)
    lines = ["Mode    Freq (cm^-1)    Reduced mass (amu)"]
    lines.append("-" * 48)
    for k, (nu, m) in enumerate(zip(freqs, mu), start=1):
        flag = " (i)" if nu < 0 else "    "
        lines.append(f"{k:4d}    {nu:12.4f}{flag}      {m:10.4f}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Transform table (replaces x_to_q.py's "transform_cartesian_normal" output)
# -----------------------------------------------------------------------------
def _write_matrix(handle: TextIO, name: str, M: np.ndarray) -> None:
    handle.write(f"{name}\n")
    for row in M:
        handle.write("".join(f"{v:20.8f}" for v in row))
        handle.write("\n")
    handle.write("\n\n")


def _write_vector(handle: TextIO, name: str, v: np.ndarray) -> None:
    handle.write(f"{name}\n")
    handle.write("".join(f"{x:20.8f}" for x in v))
    handle.write("\n\n")


def write_transform_table(
    path: str | Path,
    molecule: Molecule,
    analysis: VibrationalAnalysis,
    transform: NormalCoordinateTransform,
    *,
    dimensionless: bool = False,
) -> None:
    """Write the cartesian <-> normal coordinate transformation matrices
    in the same layout your old ``transform_cartesian_normal`` file used.
    """
    natom = molecule.natom
    nmode = analysis.nmode
    masses = np.asarray(molecule.masses)
    coords = np.asarray(molecule.coordinates)
    freqs = np.asarray(analysis.frequencies_cm)
    L = np.asarray(analysis.modes_mw_cart)

    if dimensionless:
        Q_to_x = np.asarray(transform.q_to_x)
        x_to_Q = np.asarray(transform.x_to_q)
    else:
        Q_to_x = np.asarray(transform.Q_to_x)
        x_to_Q = np.asarray(transform.x_to_Q)

    with open(path, "w") as h:
        h.write(f"{natom}\n")
        h.write("".join(f"{m:20.8f}" for m in masses))
        h.write("\n")
        h.write(f"{nmode}\n")
        _write_matrix(h, "Cartesian Reference Geometry", coords)
        _write_matrix(h, "Transformation Matrix for Q to X", Q_to_x)
        _write_matrix(h, "Transformation Matrix for X to Q", x_to_Q)
        _write_matrix(h, "Reference Normal Modes", L)
        _write_vector(h, "Reference Frequencies", freqs)


# -----------------------------------------------------------------------------
# Molden format — for visualizing modes in Avogadro, Molden, Jmol, etc.
# -----------------------------------------------------------------------------
def write_molden(
    path: str | Path,
    molecule: Molecule,
    analysis: VibrationalAnalysis,
) -> None:
    """Write a Molden-format ``.mold`` file containing the geometry and
    normal modes. Avogadro / Molden / Jmol can visualise these directly.
    """
    symbols = molecule.symbols
    coords_ang = np.asarray(molecule.coordinates) * BOHR_TO_ANG
    freqs = np.asarray(analysis.frequencies_cm)
    L_cart = np.asarray(analysis.modes_cart)  # (3N, nmode)
    natom = molecule.natom

    with open(path, "w") as f:
        f.write("[Molden Format]\n")
        f.write("[FREQ]\n")
        for nu in freqs:
            f.write(f"{nu:14.4f}\n")
        f.write("[FR-COORD]\n")
        # Molden's coord block is in Bohr in older versions; the [FR-COORD]
        # block specifically is in Bohr. Use Bohr directly.
        coords_bohr = np.asarray(molecule.coordinates)
        for sym, xyz in zip(symbols, coords_bohr):
            f.write(f"{sym:<2s} {xyz[0]:14.6f} {xyz[1]:14.6f} {xyz[2]:14.6f}\n")
        f.write("[FR-NORM-COORD]\n")
        for k in range(analysis.nmode):
            f.write(f"vibration {k+1}\n")
            disp = L_cart[:, k].reshape(natom, 3)
            for xyz in disp:
                f.write(f"{xyz[0]:14.6f} {xyz[1]:14.6f} {xyz[2]:14.6f}\n")


# -----------------------------------------------------------------------------
# XYZ output (handy for sanity checking the geometry)
# -----------------------------------------------------------------------------
def write_xyz(path: str | Path, molecule: Molecule, comment: str = "") -> None:
    """Write a standard XYZ file (Angstrom)."""
    coords_ang = np.asarray(molecule.coordinates) * BOHR_TO_ANG
    with open(path, "w") as f:
        f.write(f"{molecule.natom}\n{comment}\n")
        for sym, xyz in zip(molecule.symbols, coords_ang):
            f.write(f"{sym:<2s} {xyz[0]:14.6f} {xyz[1]:14.6f} {xyz[2]:14.6f}\n")


__all__ = ["format_frequencies", "write_transform_table", "write_molden", "write_xyz"]
