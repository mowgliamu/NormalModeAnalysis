# nma_jax — Normal-Mode Analysis in JAX

A modernised, JAX-based rewrite of an old normal-mode-analysis package.
Reads Gaussian fchk/log output, projects out translation and rotation, diagonalises
the mass-weighted Hessian, and produces frequencies, normal modes,
RRHO thermochemistry, Duschinsky rotations, and Molden visualisation files.

The original package was written for Python 2 and `numpy`. This version
targets Python 3.10+, uses `jax` for the numerics, ships proper type hints
and tests, and fixes a handful of bugs that had crept in over time.

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

## What changed vs. the original code

### Bugs fixed

1. **Translations were never projected from the gradient.**
   The old code stored only the rotational basis on `self.Ltransrot`
   but the driver treated it as a 3N × 6 matrix, so gradient projection
   silently ignored the three translational directions.
2. **`project_translation` and `project_rotation` flags were ignored.**
   The input parser read them but the analysis never used them; both
   were always projected. They now actually take effect.
3. **Hessian eigenvalue indexing was off by 3.** `np.diagflat(rphval[3:])`
   skipped translations but included rotations.
4. **Linear molecules were not supported** (TODO in the original).
   They now are: we detect linearity, drop the redundant rotation column
   via SVD before orthonormalising, and emit 3N − 5 frequencies.
5. **Imaginary frequencies were collapsed with `abs()`,** destroying the
   information you need to identify the TS mode. They are now returned
   as **negative** numbers (Gaussian convention).
6. **Moment-of-inertia off-diagonals had the wrong sign.** This
   didn't affect eigenvectors (because the tensor is still
   diagonalisable) but would have given wrong rotational constants.
7. **`read_log_gaussian` returned a tuple *or* a scalar** depending on
   what was in the log, and the driver always tried to unpack two
   values. The log reader now returns a consistent `LogEnergies`
   dataclass.
8. **`@property` was used on methods that mutate state** and return
   tuples — both are non-Pythonic and were causing subtle bugs.
9. **File handles were not closed on error paths.** All file I/O now
   uses `with` blocks.
10. **Module import had global side effects** (`np.set_printoptions`).
    Gone.

### Features added

- **Thermochemistry** (RRHO ZPE / U / H / G / S / Cv / Cp), which the
  old code stubbed out but never actually implemented.
- **Duschinsky rotation** between two structures (mentioned in the old
  docstring but only the diagonal self-Duschinsky was ever computed).
- **Molden + XYZ output** for visualisation.
- A proper `nma-jax` **CLI** with `minimum` and `ts` subcommands.
- Type hints throughout.
- A test suite (`pytest tests/`).

### Architecture changes

- Data classes (`Molecule`, `VibrationalAnalysis`, `Thermochemistry`,
  `NormalCoordinateTransform`, `DuschinskyResult`, `AnalysisBundle`,
  `LogEnergies`, `FchkData`) are **frozen** and, where it makes sense,
  registered as **JAX PyTrees**. So you can `jax.jit` a function that
  takes a `Molecule` and returns a derived quantity without any
  ceremony.
- The hot numerical loops are vectorised. Where the old code had triple
  nested Python `for` loops, you'll find single matrix products or
  `einsum` calls.
- **64-bit floats** are enabled in `nma_jax/__init__.py` (chemistry
  needs the precision; 32-bit is not sufficient).
- Units are stated **once** at the dataclass level (Bohr, amu, Hartree
  internally) and not repeated. Conversion constants are re-exported
  for user convenience.
- I/O and computation are cleanly separated. The old `x_to_q.py` did
  both; that work now lives in `transformations.py` (computation) and
  `output.py` (file writing).

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
