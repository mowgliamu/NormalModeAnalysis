"""End-to-end example: load a Gaussian calculation, run NMA, write output.

Run from the repository root with one of the bundled Gaussian fchk files
(or any of your own):

    python examples/example_usage.py path/to/water.fchk           # minimum
    python examples/example_usage.py path/to/ts.fchk    --ts      # transition state
    python examples/example_usage.py path/to/water.fchk --molden out.molden

What it does
------------
1. Loads geometry, Hessian and (optional) gradient from a Gaussian fchk file.
   A matching .log file is read for energy/ZPE cross-checks if present.
2. Runs the harmonic analysis: translation/rotation are projected out, the
   mass-weighted Hessian is diagonalised, and frequencies are returned in
   cm^-1 (negative values denote imaginary modes).
3. Computes RRHO thermochemistry at the requested temperature and pressure.
4. Writes a Molden-format vibration file and an XYZ of the geometry, both
   suitable for visualisation in e.g. Avogadro, Jmol or Molden itself.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import nma_jax as nma
from nma_jax.constants import HARTREE_TO_KCALMOL, HARTREE_TO_KJMOL


def _print_summary(bundle: nma.driver.AnalysisBundle) -> None:
    """Pretty-print the headline numbers from an AnalysisBundle."""
    mol = bundle.molecule
    ana = bundle.analysis
    thermo = bundle.thermo

    print(f"Molecule:           {mol!r}")
    print(f"Linear?             {ana.is_linear}")
    print(f"Cartesian DOFs:     {mol.ndof}")
    print(f"Vibrational modes:  {ana.nmode}")
    print(f"Imaginary modes:    {bundle.n_imaginary}")
    if mol.energy is not None:
        print(f"E (Hartree):        {mol.energy:.8f}")
    if mol.zpe is not None:
        print(f"ZPE from Gaussian:  {mol.zpe:.6f} Hartree")

    print()
    print(nma.format_frequencies(ana))

    if thermo is not None:
        print()
        print(f"--- Thermochemistry at T = {thermo.temperature:g} K, "
              f"p = {thermo.pressure:g} Pa ---")
        print(f"ZPE                  : {thermo.zpe:14.8f} Hartree "
              f"({thermo.zpe * HARTREE_TO_KCALMOL:8.3f} kcal/mol)")
        print(f"Thermal corr. (U)    : {thermo.thermal_correction_to_energy:14.8f} Hartree")
        print(f"Thermal corr. (H)    : {thermo.thermal_correction_to_enthalpy:14.8f} Hartree")
        print(f"Thermal corr. (G)    : {thermo.thermal_correction_to_gibbs:14.8f} Hartree")
        print(f"S(total)             : {thermo.s_total * HARTREE_TO_KJMOL * 1000:14.4f} J/mol/K")
        print(f"Cp(total)            : {thermo.cp_total * HARTREE_TO_KJMOL * 1000:14.4f} J/mol/K")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fchk", type=Path, help="Path to Gaussian .fchk file")
    parser.add_argument("--log", type=Path, default=None,
                        help="Optional matching .log file (for energy/ZPE cross-check)")
    parser.add_argument("--ts", action="store_true",
                        help="Treat input as a transition state (expects 1 imaginary mode)")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--pressure", type=float, default=101325.0)
    parser.add_argument("--symmetry-number", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Empirical frequency scaling factor (e.g. 0.98)")
    parser.add_argument("--molden", type=Path, default=None,
                        help="Write a Molden file with the normal modes here")
    parser.add_argument("--xyz", type=Path, default=None,
                        help="Write the geometry as XYZ here")
    parser.add_argument("--transform-table", type=Path, default=None,
                        help="Dump the Cartesian <-> normal-coord transformation matrices")
    args = parser.parse_args()

    if not args.fchk.exists():
        parser.error(f"fchk file does not exist: {args.fchk}")

    # 1. Run the analysis ------------------------------------------------------
    analyse = nma.analyse_transition_state if args.ts else nma.analyse_minimum
    bundle = analyse(
        args.fchk,
        log_path=args.log,
        temperature=args.temperature,
        pressure=args.pressure,
        symmetry_number=args.symmetry_number,
        scale=args.scale,
    )

    # 2. Print the headline numbers --------------------------------------------
    _print_summary(bundle)

    # 3. Optional outputs ------------------------------------------------------
    if args.molden is not None:
        nma.write_molden(args.molden, bundle.molecule, bundle.analysis)
        print(f"\nMolden file written to {args.molden}")

    if args.xyz is not None:
        nma.write_xyz(args.xyz, bundle.molecule)
        print(f"XYZ file written to {args.xyz}")

    if args.transform_table is not None:
        transforms = nma.build_transforms(bundle.molecule, bundle.analysis)
        nma.write_transform_table(
            args.transform_table, bundle.molecule, bundle.analysis, transforms
        )
        print(f"Transform table written to {args.transform_table}")

    # 4. Programmatic example: Duschinsky rotation between two structures ------
    # In real use you would have two distinct Gaussian calculations (e.g. S0 vs
    # S1 minimum) and want to project one set of normal modes onto the other.
    # Here we just demo the API by computing the trivial self-Duschinsky.
    dusch = nma.duschinsky(bundle.molecule, bundle.analysis,
                           bundle.molecule, bundle.analysis)
    deviation_from_identity = float(
        np.max(np.abs(np.asarray(dusch.J) - np.eye(bundle.analysis.nmode)))
    )
    print(f"\nSelf-Duschinsky |J - I|_inf = {deviation_from_identity:.2e} "
          "(should be ~machine epsilon)")


if __name__ == "__main__":
    main()
