"""Tests for the ORCA frequency-job parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

from nma_jax import read_hess, read_out, read_orca, Molecule


# ----------------------------------------------------------------------
# Synthetic water .hess
# ----------------------------------------------------------------------

# A small but format-faithful synthetic ORCA .hess for water.  The Hessian
# numbers are dummies (we test parsing, not chemistry, here); the geometry
# and masses are real (Bohr, amu).
WATER_HESS = """
$orca_hessian_file

$act_atom
0

$act_coord
0

$act_energy
-76.41234567

$hessian
9
                  0          1          2          3          4          5
      0     0.500000   0.000000   0.000000  -0.250000   0.000000   0.100000
      1     0.000000   0.400000   0.000000   0.000000  -0.200000   0.000000
      2     0.000000   0.000000   0.300000   0.080000   0.000000  -0.150000
      3    -0.250000   0.000000   0.080000   0.150000   0.000000  -0.050000
      4     0.000000  -0.200000   0.000000   0.000000   0.110000   0.000000
      5     0.100000   0.000000  -0.150000  -0.050000   0.000000   0.090000
      6    -0.250000   0.000000  -0.080000   0.100000   0.000000   0.050000
      7     0.000000  -0.200000   0.000000   0.000000   0.090000   0.000000
      8    -0.100000   0.000000  -0.150000   0.050000   0.000000   0.060000
                  6          7          8
      0    -0.250000   0.000000  -0.100000
      1     0.000000  -0.200000   0.000000
      2    -0.080000   0.000000  -0.150000
      3     0.100000   0.000000   0.050000
      4     0.000000   0.090000   0.000000
      5     0.050000   0.000000   0.060000
      6     0.150000   0.000000  -0.050000
      7     0.000000   0.110000   0.000000
      8    -0.050000   0.000000   0.090000

$vibrational_frequencies
9
   0       0.000000
   1       0.000000
   2       0.000000
   3       0.000000
   4       0.000000
   5       0.000000
   6    1595.000000
   7    3657.000000
   8    3756.000000

$atoms
3
 O   15.9994       0.000000     0.000000     0.225333
 H    1.0078       0.000000     1.442478    -0.901333
 H    1.0078       0.000000    -1.442478    -0.901333

$actual_temperature
    0.0000

$frequency_scale_factor
    1.0000
"""

# A tiny ORCA .out snippet - only the lines our parser cares about.
WATER_OUT = textwrap.dedent("""\
    *****************
    * O   R   C   A *
    *****************

    Program Version 5.0.4 -  RELEASE  -

    ... (lots of guff omitted) ...

    -----------------
    ENERGY COMPONENTS
    -----------------
    FINAL SINGLE POINT ENERGY      -76.412345670000

    ... (more guff) ...

    --------------------------
    THERMOCHEMISTRY AT 298.15K
    --------------------------

    Zero point energy                ...      0.02134567 Eh      13.39 kcal/mol

    ... (rest of the file) ...
    """)


@pytest.fixture
def water_files(tmp_path: Path) -> tuple[Path, Path]:
    """Write the synthetic water .hess and .out to a tmpdir."""
    hess_path = tmp_path / "water.hess"
    out_path = tmp_path / "water.out"
    hess_path.write_text(WATER_HESS)
    out_path.write_text(WATER_OUT)
    return hess_path, out_path


# ----------------------------------------------------------------------
# .hess parsing
# ----------------------------------------------------------------------


def test_read_hess_returns_correct_shapes(water_files):
    hess_path, _ = water_files
    r = read_hess(hess_path)
    assert r.n_atoms == 3
    assert r.symbols == ["O", "H", "H"]
    assert r.masses.shape == (3,)
    assert r.coords.shape == (3, 3)
    assert r.hessian.shape == (9, 9)


def test_read_hess_masses_and_coords(water_files):
    hess_path, _ = water_files
    r = read_hess(hess_path)
    # Masses (amu)
    np.testing.assert_allclose(r.masses, [15.9994, 1.0078, 1.0078])
    # O on z axis, H atoms symmetric about y=0
    assert abs(r.coords[0, 0]) < 1e-12
    assert abs(r.coords[0, 1]) < 1e-12
    np.testing.assert_allclose(r.coords[1, 1], -r.coords[2, 1])
    np.testing.assert_allclose(r.coords[1, 2], r.coords[2, 2])


def test_read_hess_hessian_is_symmetric(water_files):
    """The Hessian we wrote is symmetric; parsing must preserve symmetry."""
    hess_path, _ = water_files
    r = read_hess(hess_path)
    np.testing.assert_allclose(r.hessian, r.hessian.T, atol=1e-12)


def test_read_hess_specific_matrix_elements(water_files):
    """Spot-check that the column-block parsing aligned indices correctly."""
    hess_path, _ = water_files
    r = read_hess(hess_path)
    # Diagonal of first 3x3 block
    assert r.hessian[0, 0] == 0.5
    assert r.hessian[1, 1] == 0.4
    assert r.hessian[2, 2] == 0.3
    # Cross-block element (row 0, col 6 = second block, first column)
    assert r.hessian[0, 6] == -0.25
    # And its mirror via symmetry
    assert r.hessian[6, 0] == -0.25


def test_read_hess_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_hess(tmp_path / "does-not-exist.hess")


def test_read_hess_missing_atoms_block_raises(tmp_path):
    f = tmp_path / "bad.hess"
    f.write_text("$orca_hessian_file\n\n$hessian\n1\n  0\n   0  0.0\n")
    with pytest.raises(ValueError, match="atoms"):
        read_hess(f)


# ----------------------------------------------------------------------
# .out parsing
# ----------------------------------------------------------------------


def test_read_out_extracts_energy_zpe_version(water_files):
    _, out_path = water_files
    r = read_out(out_path)
    assert r.energy == pytest.approx(-76.41234567)
    assert r.zpe_unscaled == pytest.approx(0.02134567)
    assert r.version == "5.0.4"


def test_read_out_handles_missing_zpe(tmp_path):
    """A frequency-less single-point output has no ZPE; that's fine."""
    f = tmp_path / "sp.out"
    f.write_text("Program Version 5.0.4\nFINAL SINGLE POINT ENERGY -76.0\n")
    r = read_out(f)
    assert r.energy == pytest.approx(-76.0)
    assert r.zpe_unscaled is None


def test_read_out_raises_when_energy_missing(tmp_path):
    f = tmp_path / "bad.out"
    f.write_text("Program Version 5.0.4\n")
    with pytest.raises(ValueError, match="FINAL SINGLE POINT ENERGY"):
        read_out(f)


# ----------------------------------------------------------------------
# Full Molecule assembly via read_orca
# ----------------------------------------------------------------------


def test_read_orca_builds_molecule(water_files):
    hess_path, out_path = water_files
    mol = read_orca(hess_path, out_path)
    assert isinstance(mol, Molecule)
    assert mol.n_atoms == 3
    assert mol.symbols == ["O", "H", "H"]
    assert mol.hessian is not None
    assert mol.hessian.shape == (9, 9)
    assert mol.energy == pytest.approx(-76.41234567)
    assert mol.zpe_unscaled == pytest.approx(0.02134567)


def test_read_orca_without_out_file(water_files):
    """A .hess on its own is enough to run NMA - energy/ZPE just stay None."""
    hess_path, _ = water_files
    mol = read_orca(hess_path)
    assert mol.energy is None
    assert mol.zpe is None
    # Hessian still present, so NMA should run
    modes = mol.normal_modes()
    # 3 atoms, non-linear -> 3 vibrational modes
    assert modes.n_modes == 3


def test_read_orca_applies_zpe_scaling(water_files):
    hess_path, out_path = water_files
    mol = read_orca(hess_path, out_path, zpe_scaling=0.965)
    assert mol.zpe == pytest.approx(0.02134567 * 0.965)
