# nma_jax — JAX-powered normal-mode analysis

A modern, professional rewrite of a PhD-era Python-2 normal-mode-analysis
package. The numerics are all in **JAX** (jit-compiled, float64), the public
API is built around frozen dataclasses, and several long-standing bugs from
the original code have been fixed. New features include support for linear
molecules, RRHO thermochemistry, mass-weighted Eckart alignment,
Duschinsky-rotation analysis, an XYZ animation exporter, and a `nma_jax` CLI.

[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![JAX](https://img.shields.io/badge/JAX-≥0.4.20-orange)]()

## Why `nma_jax`?

A short list of things that distinguish this package from the dozen-or-so
existing normal-mode tools (Gaussian's internal NMA, ORCA's, ASE's
`Vibrations`, Psi4's, PyVibMS, etc.):

- **Differentiable end-to-end.** The numerical core is pure JAX. Every operation —
  COM-shifting, mass-weighting, the Hessian eigendecomposition, the
  trans/rot projection — is `jit`-compilable and traceable through `jax.grad`,
  `jax.vmap`, and `jax.jacrev`. You can back-propagate frequencies through
  an NMA pipeline, which makes the package directly composable with neural-network
  potentials, force-field fitting, and any other gradient-based workflow.
- **Float64 by default.** Many JAX scientific packages quietly use float32 and
  lose eigenvalue precision on stiff Hessians; `nma_jax` enables `jax_enable_x64`
  on import so this can't bite you.
- **First-class Duschinsky and Eckart.** The Duschinsky rotation matrix `J` and
  displacement vector `K`, and mass-weighted Eckart alignment via Kabsch–Umeyama
  with the `det = +1` correction, are real library calls — not buried in
  example scripts or left as exercises. Most other packages either omit these
  or hide them behind clunky text-file interfaces.
- **Multiple QC backends.** Reads Gaussian (`.log` + `.fchk`) **and**
  ORCA (`.hess` + `.out`); the parsers return strongly-typed result objects so
  you never get a variable-length tuple back. Adding Psi4 or Q-Chem is ~150 lines
  using `io_orca.py` as a template.
- **Linear molecules done right.** `is_linear` detects the geometry and the
  analysis cleanly drops to $3N-5$ modes. Rotational constants flag the
  $A$ axis as `inf` instead of silently mis-reporting.
- **RRHO thermochemistry built in.** Sackur–Tetrode translational, asymmetric-top
  and linear rotational, harmonic vibrational (with imaginary-mode handling for
  transition states), and electronic contributions to $U$, $H$, $S$, and $G$ at
  arbitrary $T$ and $P$.
- **Frozen-dataclass API.** `Molecule`, `NormalModes`, `Thermo`,
  `EckartResult`, `DuschinskyResult`, and `TransformationMatrices` are immutable.
  Every transformation (`shift_to_com`, `to_principal_frame`,
  `apply_eckart_rotation`, …) returns a new instance. No surprise mutations,
  no `@property`-with-side-effects anti-patterns.
- **Imaginary frequencies as negative reals.** No sentinel values, no
  bookkeeping; transition states just work, and `n_imaginary` is one attribute
  away.
- **Includes a CLI and an XYZ animation exporter.** `nma analyze freq.log freq.fchk
  --thermo` for quick inspection without writing a script; `modes.write_xyz_animation(...)`
  to visualise any mode in VMD or Molden.
- **A worked example notebook.** [`examples/duschinsky_franck_condon_water.ipynb`](examples/duschinsky_franck_condon_water.ipynb)
  goes from two states to a broadened Franck–Condon spectrum in twenty cells.
- **30 unit tests, fully passing.** Covers sign conventions, projectors,
  linear and non-linear molecules, gradient projection (the original code's
  Achilles' heel), Eckart alignment, Duschinsky round-trips, transformation-matrix
  inverses, thermochemistry, and CLI plumbing.

## Install

```bash
pip install -e .
# optional dev tools
pip install -e ".[dev]"
pytest
```

The package requires Python ≥ 3.10 and JAX ≥ 0.4.20. `float64` is enabled
automatically inside `nma_jax/__init__.py`.

## Quick start

```python
from nma_jax import Molecule, read_gaussian, thermochemistry

# Load a Gaussian frequency calculation
mol = read_gaussian("freq.log", "freq.fchk")

# ... or an ORCA frequency calculation
# from nma_jax import read_orca
# mol = read_orca("freq.hess", "freq.out")

# Run harmonic vibrational analysis
modes = mol.normal_modes(project_translation=True, project_rotation=True)
print(modes.table())                 # human-readable table

# Inspect rotational constants
print("Rotational constants (cm^-1):", mol.rotational_constants_cm)

# Standard RRHO thermochemistry at T = 298.15 K
thermo = thermochemistry(modes, temperature=298.15, symmetry_number=2)
print(thermo)

# Export a VMD-readable animated trajectory of mode 0
modes.write_xyz_animation("mode0.xyz", mode_index=0, amplitude=0.5)
```

Comparing two geometries (Eckart alignment + Duschinsky matrix):

```python
from nma_jax import read_gaussian, eckart_align, duschinsky

ref = read_gaussian("min.log", "min.fchk")
disp = read_gaussian("disp.log", "disp.fchk")

eck = eckart_align(ref, disp)
print(f"Mass-weighted RMSD after alignment: {eck.rmsd:.6f} Bohr")

dus = duschinsky(ref.normal_modes(), disp.normal_modes())
# dus.J: (nmode, nmode) Duschinsky rotation
# dus.K: (nmode,) displacement vector along displaced normal coordinates
```

Command-line:

```bash
nma-jax analyze freq.log freq.fchk --thermo --temperature 298.15 --sigma 2
nma-jax compare ref.log ref.fchk disp.log disp.fchk
```

## Public API

| Object | Module | Purpose |
| --- | --- | --- |
| `Molecule` | `nma.molecule` | Frozen dataclass holding atoms, masses, coords, optional Hessian / gradient / energy / ZPE |
| `NormalModes` | `nma.molecule` | Result of `Molecule.normal_modes()` — frequencies, mwc and Cartesian modes, normal-coordinate gradient, etc. |
| `Thermo` | `nma.thermo` | RRHO thermochemistry values (ZPE, U, H, G, S split into trans/rot/vib/elec) |
| `EckartResult` | `nma.eckart` | Output of `eckart_align()`: proper rotation + mass-weighted RMSD |
| `DuschinskyResult` | `nma.eckart` | Output of `duschinsky()`: J matrix, K displacement, Eckart rotation |
| `TransformationMatrices` | `nma.transforms` | Q ↔ X transforms in mass-weighted and dimensionless-frequency-scaled bases |
| `LogResult`, `FchkResult` | `nma.io_gaussian` | Strongly-typed Gaussian parser outputs |

`nma.core` exposes the pure-JAX building blocks (`shift_to_com`,
`inertia_tensor`, `translation_vectors`, `rotation_vectors`,
`project_out_trans_rot`, `mass_weight_hessian`, `harmonic_analysis`, …) for
users who want to compose their own pipeline.

## What changed vs the original

### Bug fixes

| # | Bug in original | Fix |
| --- | --- | --- |
| 1 | **Wrong sign on inertia-tensor off-diagonals** — used `+ m·xy` instead of `−m·xy`. Silently gave a wrong tensor whenever the molecule wasn't already in principal-axis orientation. | Correct sign in `core.inertia_tensor`; regression test `test_inertia_tensor_sign_convention`. |
| 2 | Moment of inertia computed about the origin, not the COM. `mwcart` was never re-shifted, and `get_eckart_frame_self` was decorated `@property` (which would have run on access) but never accessed. | `core.inertia_tensor` consumes COM-shifted coordinates; `Molecule.inertia_tensor` shifts first. |
| 3 | `self.Ltransrot = Drot` saved **only the rotation** part of the trans+rot basis, so the gradient projection in the driver removed only rotation, not translation. | Full trans+rot projector basis is returned and used (`harmonic_analysis(...)["projector_basis"]`). Regression tests cover both pure-translation and pure-rotation gradients. |
| 4 | `self.hessian = np.diagflat(rphval[3:])` — wrong slice (started at index 3, which mixed three zero modes into the vibrational block). | The reduced normal-mode Hessian is just `diag(eigenvalues)` of the vibrational subspace. |
| 5 | `read_log_gaussian` returned **variable-length tuples** (1- or 2-tuple depending on whether ZPE was found), which crashed the driver when unpacking. | `read_log` returns a typed `LogResult` (always the same shape). |
| 6 | File handles leaked: `re.search(open(filename).read())` never closes the file. | Uses `with open(...)` everywhere. |
| 7 | `@property` decorators on methods with side effects (`get_eckart_frame_self`, `harmonic_vibrational_analysis`). Mutating state on attribute access is a well-known anti-pattern. | All transformations return new immutable instances (`replace(self, ...)`). |
| 8 | ZPE scaling factor was mentioned in a comment but never applied. | `Molecule.zpe` returns `zpe_unscaled * zpe_scaling`; the scaling can be passed on construction or set per-call. |
| 9 | Linear molecules unsupported (a TODO in the source). | `core.is_linear` detects them and `harmonic_analysis` projects out `3N−5` modes accordingly. Test on N₂. |
| 10 | `hfreq_cm` defined twice (`input_nma.py` *and* `physical_constants.py`), at different precisions. | One CODATA-2018-derived constant, `nma.constants.HFREQ_CM`. |
| 11 | `get_centre_of_mass_mwc` did a `sqrt(m)·sqrt(m)·x` detour instead of plain `m·x`. | `core.center_of_mass` is the obvious one-liner. |
| 12 | Class attributes declared but never implemented: `PointGroup`, `RotationalConstants`, `Duschinsky`, `EckartMatrix`, `Displacement`. | `rotational_constants_cm` / `_ghz` are now properties; `EckartMatrix` and `Duschinsky`/`Displacement` are real first-class operations (`eckart_align`, `duschinsky`). Point-group detection is left as a TODO. |

### New features

- **JAX everywhere.** Pure functions in `nma.core` are `jit`-compiled and
  differentiable. The package enables `jax_enable_x64` automatically on
  import so you don't silently lose precision.
- **Linear-molecule support** (`3N − 5` modes, infinite rotational constant
  flagged as `inf`).
- **RRHO thermochemistry** — translational (Sackur-Tetrode), rotational
  (linear and asymmetric-top branches), vibrational (with imaginary-mode
  handling), electronic; arbitrary T and P; user-settable symmetry number
  and electronic degeneracy.
- **Mass-weighted Eckart alignment** via the Kabsch–Umeyama SVD with the
  `det = +1` correction, returning the proper rotation and the mass-weighted
  RMSD.
- **Duschinsky-rotation analysis** between any two `NormalModes` — returns
  `J` and `K` so you can build harmonic Franck-Condon factors downstream.
- **CLI** (`nma analyze`, `nma compare`) — useful for quick inspection
  without writing a script.
- **XYZ animation export** of any normal mode for VMD/Molden visualisation.
- **Modern packaging.** `src/` layout, PEP 621 `pyproject.toml`, type
  annotations, `pytest` test suite, `ruff` and `mypy` config.

### Architecture

```
nma_jax/
├── __init__.py        public API, enables jax_enable_x64
├── constants.py       CODATA-2018 constants, atomic symbol table
├── core.py            pure JAX functions: COM, inertia, projectors, harmonic_analysis
├── molecule.py        Molecule + NormalModes dataclasses (high-level API)
├── io_gaussian.py     read_log, read_fchk, read_gaussian
├── io_orca.py         read_hess, read_out, read_orca
├── eckart.py          Eckart alignment + Duschinsky (J, K)
├── transforms.py      Q ↔ X transformation matrices (legacy-compatible writer)
├── thermo.py          RRHO thermochemistry
└── cli.py             nma analyze / nma compare entry point
tests/
├── test_core.py       18 tests: core math, NMA, Eckart, Duschinsky, thermo, CLI
└── test_io_orca.py    12 tests: .hess / .out parsing, Molecule assembly
examples/
└── duschinsky_franck_condon_water.ipynb     end-to-end vibronic-spectrum demo
```

The `core` module is the pure functional heart; `molecule` is a thin,
ergonomic wrapper around it. Anything you couldn't build with `core` directly
is probably a feature missing from the package.

## Conventions

- **Coordinates** are stored in Bohr, **masses** in atomic mass units (amu),
  **energies** in Hartree, **Hessians** in Hartree / Bohr², **gradients** in
  Hartree / Bohr.
- **Frequencies** are returned in cm⁻¹. Imaginary frequencies are returned
  as **negative real numbers** (so `frequencies[0] = -512.0` means
  512 i cm⁻¹).
- The `rotation` matrix from `eckart_align` is **right-multiplying**:
  `displaced.coords ≈ reference.coords @ rotation`. The Hessian
  transforms as `H' = R₃ₙᵀ H R₃ₙ` with `R₃ₙ` block-diagonal.
- All `Molecule` methods (`shift_to_com`, `to_principal_frame`, …) return
  **new** `Molecule` instances; the class is frozen.

## Testing

```bash
pytest -v
```

Covers (30 tests):

- Centre of mass for water sits on its symmetry axis
- Inertia tensor diagonal in principal frame, correct sign convention
- Linear-vs-nonlinear detection
- Translation/rotation eigenvectors are orthonormal and mutually orthogonal
- Synthetic Hessian round-trip recovers prescribed frequencies (3N−6 = 3 for water; 3N−5 = 1 for N₂)
- Eckart alignment exactly recovers an applied rotation, with the expected `coords @ rotation` convention
- Pure-translation and pure-rotation gradients both project to zero in normal-mode space (the original code only handled rotation)
- Duschinsky `J = I`, `K = 0` for identical states *and* under a rigid rotation of the displaced state
- `D^+ D = I` round-trip on the Q ↔ X transformation matrices (mass-weighted and dimensionless)
- End-to-end thermochemistry on water gives sensible positive entropies
- CLI parser builds and exposes both subcommands
- ORCA `.hess` parsing: shapes, symmetry, column-block alignment, missing-block errors
- ORCA `.out` parsing: energy, ZPE, version; graceful handling of single-point outputs
- `read_orca` end-to-end builds a working `Molecule` and applies ZPE scaling

## Known limitations / TODO

- **Point-group / symmetry assignment** is not implemented (left as a
  declared-but-empty attribute in the original; would need a symmetry-element
  detector — out of scope for this rewrite).
- Gaussian and ORCA are supported; Psi4 / Q-Chem parsers would be ~150 lines
  each using `io_orca.py` as a template.
- The Q ↔ X transformation file writer in `transforms.py` reproduces the
  original `transform_cartesian_normal` format for backwards compatibility;
  any new code should just keep the `TransformationMatrices` object around.

## License

MIT.
