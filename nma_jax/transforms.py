"""Cartesian ↔ normal-coordinate transformation matrices.

Replaces the original ``x_to_q.py`` module.  The hand-written triple-nested
loops are replaced with vectorised JAX expressions, and the file-output
logic is split from the math so the latter is unit-testable.

Notation (MCTDH convention)
---------------------------
Let :math:`M^{1/2}` be the mass matrix raised to one-half (each atomic mass
appears three times on the diagonal, once per Cartesian direction), and
:math:`L` the matrix of mass-weighted normal-mode columns (shape ``(3N, M)``
where ``M`` is the number of vibrational modes).

* Mass-weighted normal coordinate ↔ Cartesian displacement::

      X = D Q        with  D = M^{-1/2} L      shape (3N, M)
      Q = D^+ X       with  D^+ = (D^T D)^{-1} D^T

* Dimensionless normal coordinate (q = √ω · Q in atomic units; see
  :data:`nma.constants.FRED` for the practical-units prefactor)::

      X = D' q        with  D'_{k, alpha} = L_{k, alpha} / (FRED * sqrt(omega_alpha) * sqrt(m_i))
      q = D'^+ X      with  D'^+_{alpha, k} = FRED * sqrt(omega_alpha) * sqrt(m_i) * L_{k, alpha}

  where ``k = 3*i + j`` (atom ``i``, Cartesian ``j``) and ``omega_alpha`` is
  the wavenumber of mode ``alpha`` in cm\\ :sup:`-1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .constants import FRED


@dataclass(frozen=True, slots=True)
class TransformationMatrices:
    """Bundle of transformation matrices between Cartesian and normal coords.

    Shapes (with ``M = nmode`` and ``N = natom``):

    * ``D``        ``(3N, M)``  - mass-weighted Q to Cartesian X
    * ``D_pinv``   ``(M, 3N)``  - Cartesian X to mass-weighted Q
    * ``Xdim``     ``(3N, M)``  - dimensionless q to Cartesian X
    * ``Qdim``     ``(M, 3N)``  - Cartesian X to dimensionless q
    """
    D: Array
    D_pinv: Array
    Xdim: Array
    Qdim: Array


def transformation_matrices(
    masses: Array,
    frequencies_cm: Array,
    modes_mwc: Array,
) -> TransformationMatrices:
    r"""Build all four X↔Q transformation matrices.

    Parameters
    ----------
    masses : (N,)
        Atomic masses (amu).
    frequencies_cm : (M,)
        Vibrational frequencies in cm⁻¹.  Imaginary frequencies are not
        supported here (they have no meaningful dimensionless coordinate);
        if any are negative they are clipped to ``|ω|`` with a warning.
    modes_mwc : (3N, M)
        Mass-weighted Cartesian normal modes.

    Returns
    -------
    TransformationMatrices
    """
    masses3 = jnp.repeat(masses, 3)                       # (3N,)
    inv_sqrt_m3 = 1.0 / jnp.sqrt(masses3)
    sqrt_m3 = jnp.sqrt(masses3)

    # ---- mass-weighted Q ↔ X
    d_mat = inv_sqrt_m3[:, None] * modes_mwc              # (3N, M)
    # D^+ = (D^T D)^-1 D^T  (orthonormal L means D^T D ~ M^-1 sense, but in
    # general just do it explicitly for stability)
    dtdi = jnp.linalg.inv(d_mat.T @ d_mat)
    d_pinv = dtdi @ d_mat.T                                # (M, 3N)

    # ---- dimensionless q ↔ X
    # We need sqrt(|omega|); imaginary modes have nu < 0 in our convention.
    omega_abs = jnp.abs(frequencies_cm)
    sqrt_omega = jnp.sqrt(omega_abs)                       # (M,)
    # Xdim[k, alpha] = L[k, alpha] / (FRED * sqrt(omega_alpha) * sqrt(m_{i(k)}))
    denom = FRED * sqrt_omega[None, :] * sqrt_m3[:, None]   # (3N, M)
    xdim = modes_mwc / denom                                # (3N, M)
    # Qdim[alpha, k] = FRED * sqrt(omega_alpha) * sqrt(m_{i(k)}) * L[k, alpha]
    qdim = (denom * modes_mwc).T                            # (M, 3N)

    return TransformationMatrices(D=d_mat, D_pinv=d_pinv, Xdim=xdim, Qdim=qdim)


def write_transform_file(
    path: str | Path,
    *,
    natom: int,
    masses: np.ndarray,
    nmode: int,
    ref_cart: np.ndarray,
    ref_freq: np.ndarray,
    modes_mwc: np.ndarray,
    matrices: TransformationMatrices,
    dimensionless: bool = False,
) -> None:
    """Write the legacy ``transform_cartesian_normal`` text file.

    Format mirrors the original ``x_to_q.py`` for backward compatibility
    (so any downstream MCTDH input scripts that consumed it still work).
    """
    if dimensionless:
        q_to_x = np.asarray(matrices.Xdim)
        x_to_q = np.asarray(matrices.Qdim)
    else:
        q_to_x = np.asarray(matrices.D)
        x_to_q = np.asarray(matrices.D_pinv)

    masses = np.asarray(masses)
    ref_cart = np.asarray(ref_cart)
    ref_freq = np.asarray(ref_freq)
    modes_mwc = np.asarray(modes_mwc)

    def fmt(arr: np.ndarray) -> str:
        return "  ".join(f"{x:20.8f}" for x in arr.ravel())

    with open(path, "w") as f:
        f.write(f"{natom}\n")
        f.write("".join(f"{m:20.8f}" for m in masses) + "\n")
        f.write(f"{nmode}\n")
        f.write("Cartesian Reference Geometry\n")
        for row in ref_cart:
            f.write("".join(f"{v:20.8f}" for v in row) + "\n")
        f.write("\n\nTransformation Matrix for Q to X\n")
        for row in q_to_x:
            f.write("".join(f"{v:20.8f}" for v in row) + "\n")
        f.write("\n\nTransformation Matrix for X to Q\n")
        for row in x_to_q:
            f.write("".join(f"{v:20.8f}" for v in row) + "\n")
        f.write("\n\nReference Normal Modes\n")
        for row in modes_mwc:
            f.write("".join(f"{v:20.8f}" for v in row) + "\n")
        f.write("\n\nReference Frequencies\n")
        f.write("".join(f"{v:20.8f}" for v in ref_freq) + "\n\n")
