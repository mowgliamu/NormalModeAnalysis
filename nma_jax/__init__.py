"""Normal-mode analysis with JAX.

A modernised replacement for the original Python-2 PhD code, with all known
bugs fixed and several new features (linear molecules, RRHO thermochemistry,
Eckart alignment, Duschinsky rotation, mode-animation export).

Top-level convenience imports
-----------------------------
::

    from nma_jax import Molecule, read_gaussian, thermochemistry, eckart_align, duschinsky

Quick start
-----------
::

    from nma_jax import read_gaussian, thermochemistry

    mol = read_gaussian("freq.log", "freq.fchk")
    modes = mol.normal_modes()
    print(modes.table())

    thermo = thermochemistry(modes, temperature=298.15, symmetry_number=2)
    print(thermo.summary())
"""

from __future__ import annotations

# Enable 64-bit precision in JAX before anything else uses it.
# Chemistry calculations need this; the default float32 quickly degrades
# eigenvalue accuracy on stiff Hessians.
import jax as _jax

_jax.config.update("jax_enable_x64", True)

# Public API
from .molecule import Molecule, NormalModes
from .io_gaussian import (
    LogResult,
    FchkResult,
    read_log,
    read_fchk,
    read_gaussian,
)
from .io_orca import (
    OrcaHessResult,
    OrcaOutResult,
    read_hess,
    read_out,
    read_orca,
)
from .eckart import (
    EckartResult,
    DuschinskyResult,
    eckart_align,
    apply_eckart_rotation,
    duschinsky,
)
from .thermo import Thermo, thermochemistry
from .transforms import TransformationMatrices, transformation_matrices, write_transform_file

__version__ = "1.0.0"

__all__ = [
    "Molecule",
    "NormalModes",
    "LogResult",
    "FchkResult",
    "read_log",
    "read_fchk",
    "read_gaussian",
    "OrcaHessResult",
    "OrcaOutResult",
    "read_hess",
    "read_out",
    "read_orca",
    "EckartResult",
    "DuschinskyResult",
    "eckart_align",
    "apply_eckart_rotation",
    "duschinsky",
    "Thermo",
    "thermochemistry",
    "TransformationMatrices",
    "transformation_matrices",
    "write_transform_file",
    "__version__",
]
