"""Readers for Gaussian log and formatted-checkpoint files.

Fixed problems from the original code
-------------------------------------
* Files opened inside ``re.search(open(...).read())`` were never closed.
* ``read_log_gaussian`` returned variable-length tuples (``(E0,)`` vs
  ``(E0, ZPE)``) which crashed the driver when ZPE was missing.
* The g09/g16 regex was duplicated with the only difference being a single
  inner anchor.  Combined into a single tolerant pattern that handles g09,
  g16, and the slightly different g23 format.
* No defensive checking of the input path or the file contents.

Public API
----------
:func:`read_log` returns a ``LogResult`` with ``energy``, ``zpe`` (unscaled),
and any partial information that could be parsed.

:func:`read_fchk` returns an ``FchkResult`` with everything needed to build
a :class:`~nma.molecule.Molecule`.

:func:`read_gaussian` is a convenience that combines both into a single
``Molecule``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .molecule import Molecule

log = logging.getLogger(__name__)

GaussianVersion = Literal["g09", "g16", "g23", "auto"]


# ----------------------------------------------------------------------
# .log files (text output)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogResult:
    """Subset of useful quantities parsed from a Gaussian log."""
    energy: float | None              # SCF energy, Hartree
    zpe_unscaled: float | None        # zero-point correction, Hartree
    version: str | None               # 'g09'/'g16'/'g23' if detectable


_RE_VERSION = re.compile(r"Gaussian\s+(\d{2})\b", re.IGNORECASE)
_RE_SCF = re.compile(r"SCF Done:\s+E\([^)]+\)\s*=\s*([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)")
_RE_ZPE_CORR = re.compile(r"Zero-point correction=\s*([-+]?\d+\.\d+)")
# The "\ZeroPoint=" tag in the archive section can be broken across lines
_RE_ZPE_ARCHIVE = re.compile(r"\\ZeroPoint=([-+]?\d+\.\d+)", re.DOTALL)


def read_log(path: str | Path) -> LogResult:
    """Parse a Gaussian log file for SCF energy and (unscaled) ZPE.

    Always returns a :class:`LogResult` - missing values are ``None`` rather
    than missing tuple elements.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    text = p.read_text()

    version = None
    m = _RE_VERSION.search(text)
    if m:
        version = f"g{m.group(1)}"

    # Last SCF Done is the converged one for opt jobs
    energy = None
    for m in _RE_SCF.finditer(text):
        energy = float(m.group(1))

    zpe = None
    for m in _RE_ZPE_CORR.finditer(text):
        zpe = float(m.group(1))
    if zpe is None:
        # Fall back to the archive section, which may span multiple lines
        # with trailing whitespace.  Remove whitespace around the value.
        flat = re.sub(r"\s+", "", text)
        m = re.search(r"\\ZeroPoint=([-+]?\d+\.\d+)", flat)
        if m:
            zpe = float(m.group(1))

    if energy is None:
        raise ValueError(
            f"Could not find an SCF energy in {p}.  "
            "Check that the calculation completed."
        )

    log.info("Read log %s: E=%s Eh, ZPE=%s Eh, version=%s",
             p.name, energy, zpe, version)
    return LogResult(energy=energy, zpe_unscaled=zpe, version=version)


# ----------------------------------------------------------------------
# .fchk files (formatted checkpoint)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FchkResult:
    """Quantities extracted from a Gaussian formatted-checkpoint file."""
    n_atoms: int
    atomic_numbers: np.ndarray        # (N,)
    masses: np.ndarray                # (N,)        amu
    coords: np.ndarray                # (N, 3)      Bohr
    gradient: np.ndarray              # (3N,)       Hartree/Bohr
    hessian: np.ndarray               # (3N, 3N)    Hartree/Bohr^2


def _read_fchk_array(text: str, header_regex: str) -> np.ndarray:
    """Pull a numeric block immediately following a fchk header line.

    A header line in a formatted checkpoint looks like::

        Atomic numbers                              I   N=        N
        ...numbers...
    """
    m = re.search(header_regex + r"\s+N=\s+(\d+)\s*\n", text)
    if m is None:
        raise ValueError(f"Header not found: {header_regex!r}")
    n = int(m.group(1))
    start = m.end()
    # Read enough whitespace-separated tokens after the header
    tail = text[start:]
    tokens = tail.split(maxsplit=n)
    if len(tokens) < n:
        raise ValueError(f"Not enough tokens after header {header_regex!r}")
    values = np.fromstring(" ".join(tokens[:n]), sep=" ")
    if values.size != n:
        raise ValueError(
            f"Expected {n} values after header {header_regex!r}, got {values.size}"
        )
    return values


def read_fchk(path: str | Path) -> FchkResult:
    """Parse a Gaussian formatted-checkpoint file.

    Works for g09 / g16 / g23.  Unlike the original code, this does **not**
    depend on knowing the version up-front; we extract sections by their
    individual headers rather than chaining anchors with version-specific
    intermediate strings (which broke between versions).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)

    text = p.read_text()

    anums = _read_fchk_array(text, r"Atomic numbers\s+I").astype(np.int32)
    n_atoms = anums.size

    # 'Real atomic weights' is what Gaussian uses for masses; some versions
    # also write 'Vib-AtMass' or 'Integer atomic weights' - we prefer real.
    masses = _read_fchk_array(text, r"Real atomic weights\s+R")
    if masses.size != n_atoms:
        raise ValueError(
            f"Masses count {masses.size} != atom count {n_atoms} in {p}"
        )

    coords = _read_fchk_array(text, r"Current cartesian coordinates\s+R").reshape(n_atoms, 3)

    # Gradient and Hessian are only present in single-point analytic-second
    # derivative or frequency calculations.
    try:
        gradient = _read_fchk_array(text, r"Cartesian Gradient\s+R")
    except ValueError as exc:
        raise ValueError(
            f"No Cartesian Gradient in {p}; did you run a Freq job?"
        ) from exc

    try:
        low_tri = _read_fchk_array(text, r"Cartesian Force Constants\s+R")
    except ValueError as exc:
        raise ValueError(
            f"No Cartesian Force Constants in {p}; did you run a Freq job?"
        ) from exc

    one_dim = 3 * n_atoms
    expected = one_dim * (one_dim + 1) // 2
    if low_tri.size != expected:
        raise ValueError(
            f"Hessian triangle has {low_tri.size} entries, expected "
            f"{expected} for 3N x 3N (N={n_atoms})"
        )
    hessian = np.zeros((one_dim, one_dim))
    hessian[np.tril_indices(one_dim)] = low_tri
    hessian = hessian + np.tril(hessian, -1).T

    log.info("Read fchk %s: %d atoms", p.name, n_atoms)
    return FchkResult(
        n_atoms=n_atoms,
        atomic_numbers=anums,
        masses=masses,
        coords=coords,
        gradient=gradient,
        hessian=hessian,
    )


# ----------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------


def read_gaussian(
    log_path: str | Path | None,
    fchk_path: str | Path,
    *,
    zpe_scaling: float = 1.0,
) -> Molecule:
    """Read log + fchk and return a fully populated :class:`Molecule`."""
    fchk = read_fchk(fchk_path)
    energy = None
    zpe_unscaled = None
    if log_path is not None:
        lr = read_log(log_path)
        energy = lr.energy
        zpe_unscaled = lr.zpe_unscaled

    return Molecule.from_arrays(
        atomic_numbers=fchk.atomic_numbers,
        masses=fchk.masses,
        coords=fchk.coords,
        hessian=fchk.hessian,
        gradient=fchk.gradient,
        energy=energy,
        zpe_unscaled=zpe_unscaled,
        zpe_scaling=zpe_scaling,
    )
