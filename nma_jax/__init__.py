"""nma_jax — modernized normal-mode analysis with JAX.

A pythonic, jit-friendly rewrite of the original PhD-era normal-mode
analysis package. See ``examples/`` for usage.
"""
from __future__ import annotations

import jax

# Use 64-bit floats by default — chemistry needs the precision.
# This affects only arrays subsequently created via jax.numpy; it does
# not retroactively change the user's settings if they import jax first.
jax.config.update("jax_enable_x64", True)

from .constants import (
    BOHR_TO_ANG,
    ANG_TO_BOHR,
    FREQ_AU_TO_CM,
    HARTREE_TO_EV,
    HARTREE_TO_KCALMOL,
    HARTREE_TO_KJMOL,
)
from .molecule import Molecule
from .vibrational import (
    VibrationalAnalysis,
    harmonic_analysis,
    project_gradient_to_modes,
    inertia_tensor,
    translation_rotation_basis,
)
from .thermochemistry import (
    Thermochemistry,
    thermochemistry,
    zero_point_energy,
    STANDARD_PRESSURE,
)
from .duschinsky import (
    DuschinskyResult,
    duschinsky,
    eckart_rotation_matrix,
)
from .transformations import (
    NormalCoordinateTransform,
    build_transforms,
    displace_along_mode,
)
from .io_gaussian import (
    LogEnergies,
    FchkData,
    read_log,
    read_fchk,
    load_molecule,
)
from .output import (
    format_frequencies,
    write_transform_table,
    write_molden,
    write_xyz,
)
from .driver import analyse_minimum, analyse_transition_state

__version__ = "1.0.0"

__all__ = [
    # core
    "Molecule",
    "VibrationalAnalysis",
    "harmonic_analysis",
    "project_gradient_to_modes",
    "inertia_tensor",
    "translation_rotation_basis",
    # thermo
    "Thermochemistry",
    "thermochemistry",
    "zero_point_energy",
    "STANDARD_PRESSURE",
    # Duschinsky
    "DuschinskyResult",
    "duschinsky",
    "eckart_rotation_matrix",
    # transforms
    "NormalCoordinateTransform",
    "build_transforms",
    "displace_along_mode",
    # I/O
    "LogEnergies",
    "FchkData",
    "read_log",
    "read_fchk",
    "load_molecule",
    "format_frequencies",
    "write_transform_table",
    "write_molden",
    "write_xyz",
    # driver
    "analyse_minimum",
    "analyse_transition_state",
    # constants (re-exported for convenience)
    "BOHR_TO_ANG",
    "ANG_TO_BOHR",
    "FREQ_AU_TO_CM",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCALMOL",
    "HARTREE_TO_KJMOL",
]
