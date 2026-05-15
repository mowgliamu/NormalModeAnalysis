"""High-level driver routines.

These functions glue together the I/O, vibrational analysis and
thermochemistry steps and return a tidy bundle of results. They're
the replacement for the old ``driver_process_abinitio_data`` function.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np

from .io_gaussian import load_molecule
from .molecule import Molecule
from .thermochemistry import Thermochemistry, thermochemistry, zero_point_energy
from .vibrational import (
    VibrationalAnalysis,
    harmonic_analysis,
    project_gradient_to_modes,
)


@dataclass(frozen=True)
class AnalysisBundle:
    """Everything you usually want for one stationary point."""
    molecule: Molecule
    analysis: VibrationalAnalysis
    thermo: Optional[Thermochemistry]
    gradient_nm: Optional[np.ndarray]  # gradient in normal-mode basis (Hartree / sqrt(amu)*Bohr)
    n_imaginary: int


def analyse_minimum(
    fchk_path: str | Path,
    log_path: str | Path | None = None,
    *,
    temperature: float = 298.15,
    pressure: float = 101_325.0,
    symmetry_number: int = 1,
    scale: float = 1.0,
    multiplicity: int = 1,
    project_translation: bool = True,
    project_rotation: bool = True,
) -> AnalysisBundle:
    """Load a Gaussian calculation, run NMA and RRHO thermochemistry.

    Raises a warning (but does not abort) if imaginary frequencies are
    detected — use :func:`analyse_transition_state` for true TS calculations.
    """
    molecule = load_molecule(fchk_path, log_path)
    analysis = harmonic_analysis(
        molecule,
        project_translation=project_translation,
        project_rotation=project_rotation,
    )
    n_imag = int(jnp.sum(analysis.frequencies_cm < 0))
    if n_imag > 0:
        import warnings
        warnings.warn(
            f"{n_imag} imaginary frequenc{'y' if n_imag == 1 else 'ies'} found; "
            "structure is not a minimum.",
            stacklevel=2,
        )

    thermo = thermochemistry(
        molecule, analysis,
        temperature=temperature,
        pressure=pressure,
        symmetry_number=symmetry_number,
        scale=scale,
        electronic_multiplicity=multiplicity,
    )

    grad_nm = None
    if molecule.gradient_cart is not None:
        # For an optimised minimum this should be ~0; we compute it anyway.
        grad_nm = np.asarray(project_gradient_to_modes(molecule, analysis))

    return AnalysisBundle(
        molecule=molecule,
        analysis=analysis,
        thermo=thermo,
        gradient_nm=grad_nm,
        n_imaginary=n_imag,
    )


def analyse_transition_state(
    fchk_path: str | Path,
    log_path: str | Path | None = None,
    *,
    temperature: float = 298.15,
    pressure: float = 101_325.0,
    symmetry_number: int = 1,
    scale: float = 1.0,
    multiplicity: int = 1,
) -> AnalysisBundle:
    """Like :func:`analyse_minimum` but expects exactly one imaginary frequency.

    Thermochemistry is computed skipping that imaginary mode.
    """
    molecule = load_molecule(fchk_path, log_path)
    analysis = harmonic_analysis(molecule)
    n_imag = int(jnp.sum(analysis.frequencies_cm < 0))
    if n_imag == 0:
        import warnings
        warnings.warn(
            "Expected one imaginary frequency for a TS but found none.",
            stacklevel=2,
        )
    elif n_imag > 1:
        import warnings
        warnings.warn(
            f"Found {n_imag} imaginary frequencies — higher-order saddle, "
            "not a true transition state.",
            stacklevel=2,
        )

    thermo = thermochemistry(
        molecule, analysis,
        temperature=temperature,
        pressure=pressure,
        symmetry_number=symmetry_number,
        scale=scale,
        electronic_multiplicity=multiplicity,
    )

    grad_nm = None
    if molecule.gradient_cart is not None:
        grad_nm = np.asarray(project_gradient_to_modes(molecule, analysis))

    return AnalysisBundle(
        molecule=molecule,
        analysis=analysis,
        thermo=thermo,
        gradient_nm=grad_nm,
        n_imaginary=n_imag,
    )


__all__ = ["AnalysisBundle", "analyse_minimum", "analyse_transition_state"]
