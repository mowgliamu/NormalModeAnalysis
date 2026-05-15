"""The :class:`Molecule` container.

A frozen, JAX-PyTree-compatible dataclass holding everything needed for a
single normal-mode-analysis problem. Computed quantities (frequencies,
normal modes, etc.) live on a separate :class:`VibrationalAnalysis` record
returned by :func:`nma_jax.vibrational.harmonic_analysis` — keeping the
input data and the results separate makes the code easier to reason about
and lets us flow ``Molecule`` instances through ``jax.jit`` without games.

Coordinates are in **Bohr**, masses in **amu**, force constants in
**Hartree / Bohr^2**. We document this here once and don't repeat it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .periodic_table import standard_weight, symbol

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Molecule:
    """A molecule and the data needed to do a normal-mode analysis on it.

    Attributes
    ----------
    atomic_numbers : Array of shape (natom,) and integer dtype
        Atomic numbers Z of the atoms.
    masses : Array of shape (natom,), in amu
    coordinates : Array of shape (natom, 3), in Bohr
    hessian_cart : Array of shape (3*natom, 3*natom), in Hartree / Bohr^2
        Cartesian force-constant matrix.
    gradient_cart : optional Array of shape (3*natom,), in Hartree / Bohr
        Cartesian gradient. ``None`` for stationary points.
    energy : optional float, in Hartree
    zpe : optional float, in Hartree
        Zero-point energy from the upstream program. Use this for cross-
        checks; the recomputed ZPE comes from
        :func:`nma_jax.thermochemistry.zero_point_energy`.
    """

    atomic_numbers: Array
    masses: Array
    coordinates: Array
    hessian_cart: Array
    gradient_cart: Optional[Array] = None
    energy: Optional[float] = None
    zpe: Optional[float] = None

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_arrays(
        cls,
        atomic_numbers,
        coordinates,
        hessian_cart,
        masses=None,
        gradient_cart=None,
        energy: Optional[float] = None,
        zpe: Optional[float] = None,
    ) -> "Molecule":
        """Build a ``Molecule`` from plain arrays. Validates shapes and units.

        If ``masses`` is ``None`` we fall back to IUPAC standard atomic
        weights for the given atomic numbers.
        """
        atomic_numbers = jnp.asarray(atomic_numbers, dtype=jnp.int32)
        coordinates = jnp.asarray(coordinates, dtype=jnp.float64)
        hessian_cart = jnp.asarray(hessian_cart, dtype=jnp.float64)

        natom = int(atomic_numbers.shape[0])
        if coordinates.shape != (natom, 3):
            raise ValueError(
                f"coordinates must have shape ({natom}, 3); got {coordinates.shape}"
            )
        if hessian_cart.shape != (3 * natom, 3 * natom):
            raise ValueError(
                f"hessian must have shape ({3*natom}, {3*natom}); got {hessian_cart.shape}"
            )

        if masses is None:
            masses = jnp.asarray(
                [standard_weight(int(z)) for z in np.asarray(atomic_numbers)],
                dtype=jnp.float64,
            )
        else:
            masses = jnp.asarray(masses, dtype=jnp.float64)
            if masses.shape != (natom,):
                raise ValueError(
                    f"masses must have shape ({natom},); got {masses.shape}"
                )

        if gradient_cart is not None:
            gradient_cart = jnp.asarray(gradient_cart, dtype=jnp.float64).reshape(-1)
            if gradient_cart.shape != (3 * natom,):
                raise ValueError(
                    f"gradient must have shape ({3*natom},); got {gradient_cart.shape}"
                )

        # Symmetrise the Hessian (force constants are symmetric; rounding in
        # upstream programs can produce small asymmetries). Cheap and safe.
        hessian_cart = 0.5 * (hessian_cart + hessian_cart.T)

        return cls(
            atomic_numbers=atomic_numbers,
            masses=masses,
            coordinates=coordinates,
            hessian_cart=hessian_cart,
            gradient_cart=gradient_cart,
            energy=energy,
            zpe=zpe,
        )

    # ------------------------------------------------------------------ derived
    @property
    def natom(self) -> int:
        return int(self.atomic_numbers.shape[0])

    @property
    def ndof(self) -> int:
        """Number of Cartesian degrees of freedom (3N)."""
        return 3 * self.natom

    @property
    def total_mass(self) -> float:
        return float(jnp.sum(self.masses))

    @property
    def symbols(self) -> list[str]:
        return [symbol(int(z)) for z in np.asarray(self.atomic_numbers)]

    @property
    def center_of_mass(self) -> Array:
        """Cartesian COM in Bohr."""
        return jnp.einsum("i,ij->j", self.masses, self.coordinates) / jnp.sum(self.masses)

    # ------------------------------------------------------------------ transforms
    def centered(self) -> "Molecule":
        """Return a copy translated so the centre of mass is at the origin."""
        return replace(self, coordinates=self.coordinates - self.center_of_mass)

    def with_coordinates(self, coordinates: Array) -> "Molecule":
        return replace(self, coordinates=jnp.asarray(coordinates))

    def with_hessian(self, hessian_cart: Array) -> "Molecule":
        return replace(self, hessian_cart=jnp.asarray(hessian_cart))

    # ------------------------------------------------------------------ pytree
    def tree_flatten(self):
        children = (
            self.atomic_numbers,
            self.masses,
            self.coordinates,
            self.hessian_cart,
            self.gradient_cart,
        )
        aux = (self.energy, self.zpe)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        atomic_numbers, masses, coordinates, hessian_cart, gradient_cart = children
        energy, zpe = aux
        return cls(
            atomic_numbers=atomic_numbers,
            masses=masses,
            coordinates=coordinates,
            hessian_cart=hessian_cart,
            gradient_cart=gradient_cart,
            energy=energy,
            zpe=zpe,
        )

    def __repr__(self) -> str:
        formula = _hill_formula(self.symbols)
        return f"Molecule({formula}, natom={self.natom})"


def _hill_formula(symbols: list[str]) -> str:
    """Produce a Hill-system formula string (C first, then H, rest alphabetical)."""
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1

    def _fmt(sym: str) -> str:
        n = counts[sym]
        return sym if n == 1 else f"{sym}{n}"

    ordered: list[str] = []
    if "C" in counts:
        ordered.append(_fmt("C"))
    if "H" in counts:
        ordered.append(_fmt("H"))
    for sym in sorted(counts):
        if sym not in ("C", "H"):
            ordered.append(_fmt(sym))
    return "".join(ordered)


__all__ = ["Molecule"]
