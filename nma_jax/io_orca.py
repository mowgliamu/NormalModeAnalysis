"""ORCA frequency-job parser.

ORCA writes the moment-of-inertia / Hessian information to a self-contained
``.hess`` text file, and the converged SCF energy / zero-point energy to the
main ``.out`` log.  This module parses both and assembles a :class:`Molecule`.

The ``.hess`` format
====================
The file is organised as a sequence of blocks, each beginning with a
``$keyword`` marker, e.g.::

    $orca_hessian_file

    $act_atom
    1

    $hessian
    9
                     0          1          2          3          4          5
        0     0.123456  -0.234567   ...
        1     ...
        ...

    $atoms
    3
     O   15.999   0.000   0.000   0.119
     H    1.008   0.000   0.763  -0.477
     H    1.008   0.000  -0.763  -0.477

We only need the ``$hessian`` and ``$atoms`` blocks; everything else is
optional and is skipped.

Units
=====
* ``$hessian`` is Hartree / Bohr^2
* ``$atoms`` lists masses in amu and coordinates in Bohr
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ATOMIC_SYMBOLS
from .molecule import Molecule

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Result containers
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OrcaHessResult:
    """Parsed contents of an ORCA ``.hess`` file."""
    n_atoms: int
    symbols: list[str]
    masses: np.ndarray       # (N,), amu
    coords: np.ndarray       # (N, 3), Bohr
    hessian: np.ndarray      # (3N, 3N), Hartree / Bohr^2


@dataclass(frozen=True, slots=True)
class OrcaOutResult:
    """Subset of ORCA ``.out`` quantities that downstream code cares about."""
    energy: float | None             # Final SCF energy, Hartree
    zpe_unscaled: float | None       # Zero-point energy, Hartree
    version: str | None              # e.g. '5.0.4'


# ----------------------------------------------------------------------
# .hess parsing
# ----------------------------------------------------------------------


def _find_block(lines: list[str], keyword: str) -> int:
    """Return the index of the line matching ``$keyword``; raise if absent."""
    target = f"${keyword}"
    for i, line in enumerate(lines):
        if line.strip() == target:
            return i
    raise ValueError(f"Could not find '${keyword}' block in .hess file")


def _parse_hessian_block(lines: list[str], start: int) -> tuple[int, np.ndarray]:
    """Parse a block-formatted matrix.  Returns (next_line_index, matrix).

    ORCA writes the (3N x 3N) Hessian in column-blocks (default 6 wide)
    with row labels on each row and column-index header rows separating
    the blocks::

        $hessian
        9
                    0          1          2          3          4          5
            0     0.123    -0.234    ...
            1     ...
            ...
                    6          7          8
            0     ...

    """
    # First non-empty line after the marker is the matrix size
    i = start + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    n = int(lines[i].strip())
    i += 1

    mat = np.zeros((n, n))
    cols_done = 0
    while cols_done < n:
        # Skip blanks
        while i < len(lines) and not lines[i].strip():
            i += 1
        # Column-index header row
        header = lines[i].split()
        col_indices = [int(c) for c in header]
        ncols = len(col_indices)
        i += 1
        # Next `n` rows hold the matrix entries for these columns
        for row in range(n):
            tokens = lines[i].split()
            # First token is the row index; the rest are values
            row_idx = int(tokens[0])
            values = [float(t) for t in tokens[1 : 1 + ncols]]
            mat[row_idx, col_indices[0] : col_indices[-1] + 1] = values
            i += 1
        cols_done += ncols
    return i, mat


def _parse_atoms_block(lines: list[str], start: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Parse the ``$atoms`` block.

    Format::

        $atoms
        N
         <symbol>  <mass_amu>  <x_bohr>  <y_bohr>  <z_bohr>
         ...
    """
    i = start + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    n = int(lines[i].strip())
    i += 1
    symbols: list[str] = []
    masses = np.zeros(n)
    coords = np.zeros((n, 3))
    for k in range(n):
        tokens = lines[i].split()
        symbols.append(tokens[0])
        masses[k] = float(tokens[1])
        coords[k] = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
        i += 1
    return symbols, masses, coords


def read_hess(path: str | Path) -> OrcaHessResult:
    """Parse an ORCA ``.hess`` file.

    Parameters
    ----------
    path : str or Path
        Path to the ``.hess`` file written by ORCA's frequency job.

    Returns
    -------
    OrcaHessResult
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    lines = p.read_text().splitlines()

    # $atoms gives geometry + masses
    atoms_idx = _find_block(lines, "atoms")
    symbols, masses, coords = _parse_atoms_block(lines, atoms_idx)

    # $hessian gives the full Cartesian Hessian (Hartree/Bohr^2)
    hess_idx = _find_block(lines, "hessian")
    _, hessian = _parse_hessian_block(lines, hess_idx)

    n_atoms = len(symbols)
    if hessian.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError(
            f"Hessian shape {hessian.shape} doesn't match 3N x 3N with N={n_atoms}"
        )

    log.info("Read .hess %s: %d atoms, Hessian %dx%d",
             p.name, n_atoms, *hessian.shape)
    return OrcaHessResult(
        n_atoms=n_atoms, symbols=symbols, masses=masses,
        coords=coords, hessian=hessian,
    )


# ----------------------------------------------------------------------
# .out parsing
# ----------------------------------------------------------------------


_RE_VERSION = re.compile(r"Program Version\s+(\d+\.\d+\.\d+)")
# Final single-point energy (covers RHF, DFT, post-HF reference energies).
_RE_FINAL_E = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)"
)
# Zero-point energy (printed in the Thermochemistry section)
_RE_ZPE = re.compile(
    r"Zero point energy\s+\.\.\.\s+([-+]?\d+\.\d+)\s+Eh"
)


def read_out(path: str | Path) -> OrcaOutResult:
    """Parse an ORCA ``.out`` log for the final SCF energy and ZPE.

    The "FINAL SINGLE POINT ENERGY" line is the converged energy of the
    method used (RHF, DFT, CCSD(T), ...).  The ZPE is parsed from the
    Thermochemistry section; if no frequency calculation was done it
    will be ``None``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    text = p.read_text()

    version = None
    m = _RE_VERSION.search(text)
    if m:
        version = m.group(1)

    energy = None
    for m in _RE_FINAL_E.finditer(text):
        energy = float(m.group(1))

    zpe = None
    m = _RE_ZPE.search(text)
    if m:
        zpe = float(m.group(1))

    if energy is None:
        raise ValueError(
            f"Could not find 'FINAL SINGLE POINT ENERGY' in {p}. "
            "Check that the calculation completed."
        )
    log.info("Read .out %s: E=%s Eh, ZPE=%s Eh, version=%s",
             p.name, energy, zpe, version)
    return OrcaOutResult(energy=energy, zpe_unscaled=zpe, version=version)


# ----------------------------------------------------------------------
# Convenience: assemble a Molecule
# ----------------------------------------------------------------------


_SYMBOL_TO_Z = {s: i for i, s in enumerate(ATOMIC_SYMBOLS) if s}


def read_orca(
    hess_path: str | Path,
    out_path: str | Path | None = None,
    *,
    zpe_scaling: float = 1.0,
) -> Molecule:
    """Build a :class:`Molecule` from an ORCA frequency-job output.

    Parameters
    ----------
    hess_path : str or Path
        Path to the ``.hess`` file (mandatory - this is where the Hessian,
        geometry and masses live).
    out_path : str or Path, optional
        Path to the matching ``.out`` log.  Used only to pick up the
        final SCF energy and (unscaled) ZPE; the calculation can still
        be analysed without it.
    zpe_scaling : float
        Factor to apply to the ZPE (default 1.0; e.g. 0.965 for
        B3LYP/6-31G(d)).

    Returns
    -------
    Molecule
        Fully populated with atoms, masses, COM-centred coordinates,
        Hessian, and (if ``out_path`` is given) energy + ZPE.
    """
    hess = read_hess(hess_path)
    atomic_numbers = []
    for s in hess.symbols:
        if s not in _SYMBOL_TO_Z:
            raise ValueError(f"Unknown element symbol in .hess: {s!r}")
        atomic_numbers.append(_SYMBOL_TO_Z[s])

    energy = None
    zpe_unscaled = None
    if out_path is not None:
        out = read_out(out_path)
        energy = out.energy
        zpe_unscaled = out.zpe_unscaled

    return Molecule.from_arrays(
        atomic_numbers=atomic_numbers,
        masses=hess.masses,
        coords=hess.coords,
        hessian=hess.hessian,
        energy=energy,
        zpe_unscaled=zpe_unscaled,
        zpe_scaling=zpe_scaling,
    )
