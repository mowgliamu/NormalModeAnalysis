"""Coordinate transformations between the various flavors of normal modes.

All in one place, vectorized with JAX. We provide the transformation
matrices explicitly so they can be cached and re-used.

We distinguish four representations:

1. Cartesian       :math:`x`      (3N components, Bohr or Angstrom)
2. Mass-weighted   :math:`y`      :math:`y_i = \\sqrt{m_i}\\,x_i`
3. Normal modes    :math:`Q`      :math:`Q = L^T y`
4. Dimensionless   :math:`q`      :math:`q_k = \\sqrt{\\omega_k / \\hbar}\\,Q_k`

The forward/back transformations are linear maps; we return all of them.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .constants import FREQ_AU_TO_CM
from .molecule import Molecule
from .vibrational import VibrationalAnalysis, mass_weighting_matrix

Array = jax.Array


@dataclass(frozen=True)
class NormalCoordinateTransform:
    """Forward and back transformations between coord systems.

    All matrices act on flattened 3N-vectors. ``x_to_Q`` and ``Q_to_x`` use
    mass-weighted normal modes (units of sqrt(amu) * Bohr). ``x_to_q`` and
    ``q_to_x`` use the dimensionless (number-operator) coordinates.

    Attributes
    ----------
    x_to_Q : (nmode, 3N)  — Cartesian (Bohr) -> mass-weighted normal modes
    Q_to_x : (3N, nmode)  — mass-weighted normal modes -> Cartesian (Bohr)
    x_to_q : (nmode, 3N)  — Cartesian (Bohr) -> dimensionless normal modes
    q_to_x : (3N, nmode)  — dimensionless normal modes -> Cartesian (Bohr)
    reference_coordinates : (natom, 3)  — Bohr
    frequencies_cm : (nmode,) — for reference; imaginary modes are negative
    """
    x_to_Q: Array
    Q_to_x: Array
    x_to_q: Array
    q_to_x: Array
    reference_coordinates: Array
    frequencies_cm: Array


def build_transforms(
    molecule: Molecule,
    analysis: VibrationalAnalysis,
) -> NormalCoordinateTransform:
    """Compute the cartesian <-> normal coordinate transformations.

    Q -> x:   x = M^{-1/2} L_mwc Q
    x -> Q:   Q = (D^T D)^{-1} D^T x       with D = M^{-1/2} L_mwc

    Dimensionless versions scale by sqrt(omega / hbar) (in atomic units,
    by ``DIMLESS_PREFACTOR * sqrt(omega_au)`` to match the historical fred).

    Imaginary modes contribute a zero column to the dimensionless transforms
    (since sqrt(omega) is ill-defined for them) — they're still present in
    the mass-weighted transforms.
    """
    Mhalf_inv = mass_weighting_matrix(molecule.masses)  # (3N, 3N)
    L_mwc = analysis.modes_mw_cart                      # (3N, nmode)

    # ---- mass-weighted normal coordinates -------------------------------
    Q_to_x = Mhalf_inv @ L_mwc                          # (3N, nmode)
    # Left pseudo-inverse:  (D^T D)^{-1} D^T
    DTD = Q_to_x.T @ Q_to_x
    x_to_Q = jnp.linalg.solve(DTD, Q_to_x.T)            # (nmode, 3N)

    # ---- dimensionless normal coordinates -------------------------------
    # omega in atomic units: omega_au = sign(eig) * sqrt(|eig|)
    # For the dimensionless transform we want sqrt(omega) only for real modes.
    eig = analysis.eigenvalues_au
    sqrt_omega_au = jnp.where(eig > 0, jnp.sqrt(jnp.abs(eig)), 0.0)
    # Scaling vector per mode
    from .constants import DIMLESS_PREFACTOR
    scale = DIMLESS_PREFACTOR * jnp.sqrt(sqrt_omega_au)  # length nmode

    # q_k = scale_k * Q_k  =>  q = diag(scale) @ x_to_Q
    x_to_q = scale[:, None] * x_to_Q                     # (nmode, 3N)
    # Inverse scaling (zero for imaginary)
    inv_scale = jnp.where(scale > 0, 1.0 / scale, 0.0)
    q_to_x = Q_to_x * inv_scale[None, :]                 # (3N, nmode)

    return NormalCoordinateTransform(
        x_to_Q=x_to_Q,
        Q_to_x=Q_to_x,
        x_to_q=x_to_q,
        q_to_x=q_to_x,
        reference_coordinates=molecule.coordinates,
        frequencies_cm=analysis.frequencies_cm,
    )


def displace_along_mode(
    molecule: Molecule,
    analysis: VibrationalAnalysis,
    mode_index: int,
    amplitude: float = 1.0,
) -> Array:
    """Return Cartesian coordinates displaced along a single normal mode.

    The displacement is :math:`amplitude` times the Cartesian column of
    that mode (i.e. ``analysis.modes_cart[:, mode_index]``). Units of
    ``amplitude`` are therefore Bohr (or whatever the reference coords use).
    """
    direction = analysis.modes_cart[:, mode_index].reshape(molecule.natom, 3)
    return molecule.coordinates + amplitude * direction


__all__ = ["NormalCoordinateTransform", "build_transforms", "displace_along_mode"]
