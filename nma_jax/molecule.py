"""High-level :class:`Molecule` dataclass.

This is the ergonomic entry point for users.  All numerical heavy lifting is
delegated to :mod:`nma.core` which exposes pure JAX functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from . import core
from .constants import (
    ATOMIC_SYMBOLS,
    BOHR_ANGSTROM,
    HARTREE_J,
    HBAR,
    SPEED_OF_LIGHT,
    symbol_for,
)


def _as_jax(x, dtype=jnp.float64) -> Array:
    """Convert any array-like to a float64 JAX array."""
    return jnp.asarray(np.asarray(x), dtype=dtype)


@dataclass(frozen=True, slots=True)
class Molecule:
    """A molecule together with optional Hessian / gradient / energy.

    All numerical fields are stored as **JAX arrays** for compatibility with
    jit, vmap, and grad in :mod:`nma.core`.

    Parameters
    ----------
    atomic_numbers : (N,) int array
    masses : (N,) float array, amu
    coords : (N, 3) float array, Bohr
    hessian : (3N, 3N) float array, Hartree / Bohr^2; optional
    gradient : (3N,) float array, Hartree / Bohr; optional
    energy : float; total electronic energy in Hartree, optional
    zpe_unscaled : float; ZPE as reported by program (Hartree), optional
    zpe_scaling : float; factor to apply to ZPE (default 1.0)

    Notes
    -----
    The class is **frozen**: methods like :meth:`shift_to_com` return new
    instances rather than mutating.  This keeps JAX traces clean and avoids
    a class of subtle bugs (the original code had several attributes that got
    out-of-sync after in-place updates).
    """

    atomic_numbers: Array
    masses: Array
    coords: Array
    hessian: Array | None = None
    gradient: Array | None = None
    energy: float | None = None
    zpe_unscaled: float | None = None
    zpe_scaling: float = 1.0

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        atomic_numbers: Iterable[int],
        masses: Iterable[float],
        coords,
        *,
        hessian=None,
        gradient=None,
        energy: float | None = None,
        zpe_unscaled: float | None = None,
        zpe_scaling: float = 1.0,
    ) -> "Molecule":
        """Construct from any array-likes; converts to JAX float64 internally."""
        anums = jnp.asarray(np.asarray(list(atomic_numbers), dtype=np.int32))
        m = _as_jax(masses)
        c = _as_jax(coords).reshape(-1, 3)
        if m.shape[0] != c.shape[0]:
            raise ValueError(
                f"Atom count mismatch: {m.shape[0]} masses vs {c.shape[0]} coords"
            )
        h = _as_jax(hessian) if hessian is not None else None
        g = _as_jax(gradient).reshape(-1) if gradient is not None else None
        if h is not None and h.shape != (3 * m.shape[0], 3 * m.shape[0]):
            raise ValueError(
                f"Hessian shape {h.shape} != (3N, 3N) with N={m.shape[0]}"
            )
        if g is not None and g.shape != (3 * m.shape[0],):
            raise ValueError(
                f"Gradient length {g.shape[0]} != 3N with N={m.shape[0]}"
            )
        return cls(
            atomic_numbers=anums,
            masses=m,
            coords=c,
            hessian=h,
            gradient=g,
            energy=energy,
            zpe_unscaled=zpe_unscaled,
            zpe_scaling=zpe_scaling,
        )

    # ------------------------------------------------------------------
    # Trivial properties
    # ------------------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return int(self.masses.shape[0])

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    @property
    def symbols(self) -> list[str]:
        return [symbol_for(int(z)) for z in self.atomic_numbers]

    @property
    def zpe(self) -> float | None:
        """ZPE with the scaling factor applied."""
        if self.zpe_unscaled is None:
            return None
        return float(self.zpe_unscaled) * float(self.zpe_scaling)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @property
    def center_of_mass(self) -> Array:
        return core.center_of_mass(self.masses, self.coords)

    def shift_to_com(self) -> "Molecule":
        """Return a new ``Molecule`` translated so the COM is at the origin."""
        return replace(self, coords=core.shift_to_com(self.masses, self.coords))

    @property
    def inertia_tensor(self) -> Array:
        com_coords = core.shift_to_com(self.masses, self.coords)
        return core.inertia_tensor(self.masses, com_coords)

    @property
    def principal_moments(self) -> Array:
        moi, _ = core.principal_axes(self.masses, self.coords)
        return moi

    @property
    def principal_axes(self) -> Array:
        _, axes = core.principal_axes(self.masses, self.coords)
        return axes

    @property
    def is_linear(self) -> bool:
        return core.is_linear(self.masses, self.coords)

    def to_principal_frame(self) -> "Molecule":
        """Translate to COM and rotate to principal-axis frame.

        Convenient before doing Eckart alignment or visualisation.
        """
        com_coords = core.shift_to_com(self.masses, self.coords)
        _, axes = core.principal_axes(self.masses, self.coords)
        new = com_coords @ axes
        if self.hessian is not None:
            # Rotate the Hessian: H' = R_3N^T H R_3N where R_3N is block-diagonal
            r3n = jax.scipy.linalg.block_diag(*([axes] * self.n_atoms))
            new_h = r3n.T @ self.hessian @ r3n
        else:
            new_h = None
        return replace(self, coords=new, hessian=new_h)

    @property
    def rotational_constants_cm(self) -> Array:
        """Rotational constants ``(A, B, C)`` in **cm^-1**, sorted ascending.

        For a linear molecule one constant is effectively infinite and is
        returned as ``inf``; the other two are equal.
        """
        moi_amu_bohr2 = self.principal_moments              # amu * Bohr^2
        # B [cm^-1] = h / (8 pi^2 c I)
        # convert I to SI: amu * Bohr^2 -> kg m^2
        from .constants import AMU_KG, BOHR_M
        moi_si = moi_amu_bohr2 * AMU_KG * BOHR_M ** 2
        from .constants import PLANCK_H, SPEED_OF_LIGHT
        # avoid division by zero for linear molecules
        eps = 1e-50
        b_per_m = PLANCK_H / (8.0 * jnp.pi ** 2 * SPEED_OF_LIGHT * jnp.maximum(moi_si, eps))
        b_cm = b_per_m * 1e-2                                # m^-1 -> cm^-1
        # Wherever moi was effectively zero, return inf
        b_cm = jnp.where(moi_si <= eps, jnp.inf, b_cm)
        return jnp.sort(b_cm)                                # ascending

    @property
    def rotational_constants_ghz(self) -> Array:
        """Rotational constants ``(A, B, C)`` in **GHz**, sorted ascending."""
        return self.rotational_constants_cm * SPEED_OF_LIGHT * 1e-7  # cm^-1 * c[m/s] * 10^-7

    # ------------------------------------------------------------------
    # Normal-mode analysis
    # ------------------------------------------------------------------
    def normal_modes(
        self,
        *,
        project_translation: bool = True,
        project_rotation: bool = True,
    ) -> "NormalModes":
        """Run harmonic vibrational analysis and return a :class:`NormalModes`."""
        if self.hessian is None:
            raise ValueError("Cannot do normal-mode analysis without a Hessian.")
        # Always work in COM-shifted coords for analysis; this is what makes
        # the rotation projection rigorously the body-frame Sayvetz construct.
        com_coords = core.shift_to_com(self.masses, self.coords)
        result = core.harmonic_analysis(
            self.masses,
            com_coords,
            self.hessian,
            project_translation=project_translation,
            project_rotation=project_rotation,
        )
        # Gradient transformed to normal-mode basis (if available)
        grad_nm = None
        if self.gradient is not None:
            inv_sqrt_m3 = 1.0 / jnp.sqrt(jnp.repeat(self.masses, 3))
            g_mw = self.gradient * inv_sqrt_m3
            # Project out translation/rotation from gradient too
            basis = result["projector_basis"]
            if basis.shape[1]:
                g_mw = g_mw - basis @ (basis.T @ g_mw)
            grad_nm = result["modes_mwc"].T @ g_mw

        return NormalModes(
            frequencies=result["frequencies"],
            modes_mwc=result["modes_mwc"],
            modes_cart=result["modes_cart"],
            eigenvalues_mwc=result["eigenvalues_mwc"],
            gradient_normal=grad_nm,
            is_linear=result["is_linear"],
            molecule=self,
        )

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Molecule(n_atoms={self.n_atoms}, "
            f"formula='{self._formula()}', "
            f"energy={self.energy})"
        )

    def _formula(self) -> str:
        from collections import Counter
        c = Counter(self.symbols)
        # Hill order: C first, H second, then alphabetical
        ordered: list[tuple[str, int]] = []
        if "C" in c:
            ordered.append(("C", c.pop("C")))
        if "H" in c:
            ordered.append(("H", c.pop("H")))
        ordered.extend(sorted(c.items()))
        return "".join(s if n == 1 else f"{s}{n}" for s, n in ordered)

    def to_xyz(self, comment: str = "") -> str:
        """Return an XMol-style XYZ string (Angstrom, atom symbols)."""
        lines = [str(self.n_atoms), comment]
        coords_ang = np.asarray(self.coords) * BOHR_ANGSTROM
        for sym, (x, y, z) in zip(self.symbols, coords_ang):
            lines.append(f"{sym:<3s} {x: .8f}  {y: .8f}  {z: .8f}")
        return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# NormalModes result object
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalModes:
    """Result of a harmonic vibrational analysis."""

    frequencies: Array            # (nmode,) cm^-1, negative = imaginary
    modes_mwc: Array              # (3N, nmode) mass-weighted Cartesian
    modes_cart: Array             # (3N, nmode) Cartesian displacements
    eigenvalues_mwc: Array        # (nmode,) au
    gradient_normal: Array | None  # (nmode,) in mass-weighted units, if gradient was given
    is_linear: bool
    molecule: Molecule = field(repr=False)

    @property
    def n_modes(self) -> int:
        return int(self.frequencies.shape[0])

    @property
    def n_imaginary(self) -> int:
        return int((self.frequencies < 0).sum())

    @property
    def real_frequencies(self) -> Array:
        """Just the real (positive) frequencies."""
        return self.frequencies[self.frequencies > 0]

    def write_xyz_animation(
        self,
        path,
        mode_index: int,
        *,
        amplitude: float = 0.5,
        n_frames: int = 20,
    ) -> None:
        """Write an animated XYZ trajectory of one mode for VMD / Molden.

        Parameters
        ----------
        path : str or Path
            Output file path.
        mode_index : int
            Which vibrational mode (0-indexed, ascending in frequency).
        amplitude : float
            Maximum displacement scaling (dimensionless; the Cartesian-mode
            vectors are unit-norm in mass-weighted space already).
        n_frames : int
            Number of frames in one period.
        """
        mol = self.molecule
        eq = np.asarray(mol.coords) * BOHR_ANGSTROM           # (N, 3) in Å
        mode = np.asarray(self.modes_cart[:, mode_index]).reshape(mol.n_atoms, 3)
        mode = mode * BOHR_ANGSTROM
        freq = float(self.frequencies[mode_index])
        symbols = mol.symbols
        with open(path, "w") as f:
            for k in range(n_frames):
                phase = np.sin(2 * np.pi * k / n_frames)
                disp = eq + amplitude * phase * mode
                f.write(f"{mol.n_atoms}\n")
                f.write(f"mode {mode_index}  freq = {freq:+.2f} cm^-1  frame {k}\n")
                for sym, (x, y, z) in zip(symbols, disp):
                    f.write(f"{sym:<3s} {x: .8f}  {y: .8f}  {z: .8f}\n")

    def table(self) -> str:
        """Return a human-readable frequency table."""
        lines = [f"  {'#':>3s}  {'freq / cm^-1':>14s}  {'eig (au)':>14s}"]
        for i, (nu, e) in enumerate(zip(np.asarray(self.frequencies),
                                        np.asarray(self.eigenvalues_mwc))):
            marker = " (i)" if nu < 0 else ""
            lines.append(f"  {i:3d}  {nu:>14.4f}{marker}  {e: 14.6e}")
        if self.n_imaginary:
            lines.append(f"  -> {self.n_imaginary} imaginary mode(s)")
        return "\n".join(lines)
