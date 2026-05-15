# nma_jax — Normal-Mode Analysis in JAX

A modernised, JAX-based normal-mode-analysis package.
Reads Gaussian fchk/log output, projects out translation and rotation, diagonalises
the mass-weighted Hessian, and produces frequencies, normal modes,
RRHO thermochemistry, Duschinsky rotations, and Molden visualisation files.

## Quick start

```bash
pip install -e .
nma-jax minimum path/to/calc.fchk --log path/to/calc.log
# or, for a transition state:
nma-jax ts path/to/ts.fchk --log path/to/ts.log
```

Programmatic use:

```python
import nma_jax as nma

bundle = nma.analyse_minimum("water.fchk", log_path="water.log")
print(nma.format_frequencies(bundle.analysis))
print(f"ZPE = {bundle.thermo.zpe:.6f} Hartree")

# Save a Molden file you can open in Avogadro / Jmol / Molden
nma.write_molden("water.molden", bundle.molecule, bundle.analysis)
```

See `examples/example_usage.py` for a more complete walkthrough.

## What it does

For one Gaussian calculation it gives you:

- Harmonic frequencies in cm⁻¹ (imaginary modes are returned as **negative**,
  Gaussian-style; not stripped to absolute values).
- Cartesian, mass-weighted Cartesian, and dimensionless normal modes.
- Reduced masses.
- Cartesian-to-normal-coordinate transformation matrices, in both dimensioned
  and dimensionless flavours.
- RRHO thermochemistry: ZPE, U(T), H(T), G(T), S, Cv, Cp at any (T, p),
  partitioned into translational / rotational / vibrational / electronic
  contributions. Linear molecules are handled correctly.
- Optional gradient projection onto the normal-mode basis.
- Molden and XYZ output for visualisation.
- Duschinsky rotation **J** and shift vector **K** between two structures
  (e.g. for Franck-Condon / vibronic work). Uses the Eckart frame to align
  the two geometries first via a Kabsch fit.

## Layout

```
nma_jax/
├── __init__.py            # public API
├── molecule.py            # Molecule dataclass (JAX PyTree)
├── vibrational.py         # harmonic_analysis, gradient projection
├── thermochemistry.py     # RRHO thermo
├── transformations.py     # x ↔ Q transformation matrices
├── duschinsky.py          # Eckart alignment + Duschinsky rotation
├── io_gaussian.py         # fchk / log readers
├── output.py              # Molden, XYZ, transform-table writers
├── periodic_table.py      # element symbols + IUPAC atomic weights
├── constants.py           # CODATA 2018 constants
├── driver.py              # high-level analyse_minimum / analyse_transition_state
└── cli.py                 # `nma-jax` command-line tool
examples/example_usage.py
tests/test_basic.py
```

## Installation

```bash
git clone https://github.com/mowgliamu/NormalModeAnalysis.git
cd normal_mode_analysis_jax
pip install -e .
```

Requires Python ≥ 3.10, JAX ≥ 0.4, NumPy ≥ 1.22.

## Running the tests

```bash
pip install pytest
pytest tests/
```

All ten tests should pass; the geometric ones round-trip to machine
epsilon (~1e-16) on x86-64.

## Notes on conventions

- **Frequencies** are in cm⁻¹. Imaginary ones carry a minus sign.
- **Coordinates** are in **Bohr** internally; the fchk reader does the
  unit conversion for you. The XYZ writer emits Ångström.
- **Hessian** is in Hartree / Bohr².
- **Masses** are in **amu**. We use IUPAC 2021 standard atomic weights
  unless you pass an explicit `masses=` array to `Molecule.from_arrays`.
- **Modes** come out **sorted by signed eigenvalue**, so imaginary modes
  appear first. The mass-weighted modes `L = modes_mw_cart` are
  column-orthonormal; the dimensioned Cartesian modes `modes_cart` are
  the mass-weighted ones divided by √m and are *not* orthonormal — that
  is intentional and matches what most other programs print.
- **Standard pressure** is 101 325 Pa (1 atm); change it with the
  `--pressure` flag or the `pressure=` keyword.
