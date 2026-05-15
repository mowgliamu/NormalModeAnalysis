"""Read data from Gaussian output files.

Two formats are supported:

- ``.log``    plain text job output. We pull single-point energy and the
              unscaled zero-point energy (the printed ``Zero-point correction``
              line; **not** the scaled ``E(ZPE)`` line).
- ``.fchk``   formatted checkpoint, the more reliable source for arrays.
              We use a robust block-based parser that does not depend on
              Gaussian version — we detect block boundaries by name.

Coordinates come out in **Bohr**, gradients in **Hartree / Bohr**, Hessians in
**Hartree / Bohr^2**, masses in **amu**.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .molecule import Molecule


# -----------------------------------------------------------------------------
# .log: SCF and ZPE
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LogEnergies:
    scf_energy: Optional[float]   # Hartree
    zpe_unscaled: Optional[float] # Hartree


def read_log(path: str | Path) -> LogEnergies:
    """Parse a Gaussian ``.log`` for the SCF energy and (unscaled) ZPE.

    Either value may be ``None`` if the corresponding line was not in the
    file — the caller decides what to do about that. (Unlike the original
    code, we never return a bare scalar pretending to be a tuple.)
    """
    scf_energy: Optional[float] = None
    zpe: Optional[float] = None

    with open(path, "r") as handle:
        for line in handle:
            if "SCF Done:" in line:
                # Format: " SCF Done:  E(RHF) =  -76.0265...    A.U. after ..."
                try:
                    scf_energy = float(line.split()[4])
                except (IndexError, ValueError):
                    pass
            elif "Zero-point correction=" in line:
                # Format: " Zero-point correction=    0.020849 (Hartree/Particle)"
                try:
                    zpe = float(line.split()[2])
                except (IndexError, ValueError):
                    pass

    return LogEnergies(scf_energy=scf_energy, zpe_unscaled=zpe)


# -----------------------------------------------------------------------------
# .fchk: arrays
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FchkData:
    natom: int
    atomic_numbers: np.ndarray   # (natom,) int
    masses: np.ndarray           # (natom,) float, amu
    coordinates: np.ndarray      # (natom, 3) float, Bohr
    gradient: Optional[np.ndarray]    # (3*natom,) float, Hartree / Bohr
    hessian: np.ndarray          # (3N, 3N) float, Hartree / Bohr^2


# Fchk lines we care about look like:
#   Atomic numbers                             I   N=          3
#   Real atomic weights                        R   N=          3
#   Current cartesian coordinates              R   N=          9
#   Cartesian Gradient                         R   N=          9
#   Cartesian Force Constants                  R   N=         45
#
# The data follows on the next line(s), 5 floats per line (R) or 6 ints per
# line (I). We do block-based parsing instead of fragile big regexes — it
# works regardless of Gaussian version (g09, g16, g23...).
def _read_fchk_block(lines: list[str], header: str, dtype: type) -> np.ndarray | None:
    """Find ``header`` in ``lines`` and read the following array.

    Returns ``None`` if the header is not present. Headers in fchk files are
    space-padded to a fixed column width; we match by ``startswith`` after
    stripping trailing whitespace from the prefix.
    """
    target = header
    for i, line in enumerate(lines):
        if not line.startswith(target):
            continue
        # The "N=  <count>" sits at the end of the header line.
        try:
            count = int(line.split("N=")[1].split()[0])
        except (IndexError, ValueError):
            return None
        # Read enough following lines to cover `count` items
        per_line = 6 if dtype is int else 5
        nlines = (count + per_line - 1) // per_line
        chunk = lines[i + 1 : i + 1 + nlines]
        flat = " ".join(chunk).split()
        if len(flat) < count:
            raise ValueError(
                f"Truncated block '{header.strip()}' (expected {count}, got {len(flat)})"
            )
        return np.asarray(flat[:count], dtype=dtype)
    return None


def read_fchk(path: str | Path) -> FchkData:
    """Read atoms, masses, coordinates, gradient and Hessian from a fchk file.

    Works for any modern Gaussian version (g09, g16, g23+); the parser
    matches block headers by name rather than relying on the ordering of
    surrounding blocks (which differs between versions — the original
    regex was version-specific for this reason).
    """
    path = Path(path)
    with open(path, "r") as handle:
        lines = handle.readlines()

    Z = _read_fchk_block(lines, "Atomic numbers", int)
    if Z is None:
        raise ValueError(f"'Atomic numbers' block not found in {path}")
    natom = int(Z.size)

    masses = _read_fchk_block(lines, "Real atomic weights", float)
    if masses is None:
        # Older fchks may use this alternate header
        masses = _read_fchk_block(lines, "Atomic weights", float)
    if masses is None:
        raise ValueError(f"Atomic weights not found in {path}")

    coords_flat = _read_fchk_block(lines, "Current cartesian coordinates", float)
    if coords_flat is None:
        raise ValueError(f"'Current cartesian coordinates' not found in {path}")
    coords = coords_flat.reshape(natom, 3)

    grad_flat = _read_fchk_block(lines, "Cartesian Gradient", float)
    gradient = grad_flat if grad_flat is not None else None

    hess_tri = _read_fchk_block(lines, "Cartesian Force Constants", float)
    if hess_tri is None:
        raise ValueError(f"'Cartesian Force Constants' not found in {path}")

    # Gaussian stores the lower triangle row-major
    ndof = 3 * natom
    H = np.zeros((ndof, ndof), dtype=float)
    tri_idx = np.tril_indices(ndof)
    if hess_tri.size != tri_idx[0].size:
        raise ValueError(
            f"Hessian size mismatch in {path}: expected {tri_idx[0].size}, got {hess_tri.size}"
        )
    H[tri_idx] = hess_tri
    H = H + np.tril(H, -1).T  # mirror the lower triangle

    return FchkData(
        natom=natom,
        atomic_numbers=Z,
        masses=masses,
        coordinates=coords,
        gradient=gradient,
        hessian=H,
    )


# -----------------------------------------------------------------------------
# Convenience: build a Molecule directly
# -----------------------------------------------------------------------------
def load_molecule(
    fchk_path: str | Path,
    log_path: str | Path | None = None,
) -> Molecule:
    """Build a :class:`Molecule` from a Gaussian ``.fchk`` (+ optional ``.log``).

    The log file is only used to pick up the SCF energy and unscaled ZPE for
    reference — all the arrays come from the fchk.
    """
    data = read_fchk(fchk_path)

    energy: Optional[float] = None
    zpe: Optional[float] = None
    if log_path is not None:
        log = read_log(log_path)
        energy = log.scf_energy
        zpe = log.zpe_unscaled

    return Molecule.from_arrays(
        atomic_numbers=data.atomic_numbers,
        coordinates=data.coordinates,
        hessian_cart=data.hessian,
        masses=data.masses,
        gradient_cart=data.gradient,
        energy=energy,
        zpe=zpe,
    )


__all__ = ["LogEnergies", "FchkData", "read_log", "read_fchk", "load_molecule"]
