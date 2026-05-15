"""Command-line interface for nma-jax.

Usage::

    nma-jax FCHK [--log LOG] [--temperature 298.15] [--sigma 1] ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .driver import analyse_minimum, analyse_transition_state
from .output import format_frequencies, write_molden, write_xyz
from .constants import HARTREE_TO_KCALMOL, HARTREE_TO_KJMOL


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("fchk", type=Path, help="Path to the Gaussian .fchk file")
    parser.add_argument("--log", type=Path, default=None, help="Optional .log file for SCF/ZPE")
    parser.add_argument("-T", "--temperature", type=float, default=298.15, help="K")
    parser.add_argument("-P", "--pressure", type=float, default=101_325.0, help="Pa")
    parser.add_argument("--sigma", type=int, default=1, help="Symmetry number for rotation")
    parser.add_argument("--multiplicity", type=int, default=1, help="Electronic multiplicity")
    parser.add_argument("--scale", type=float, default=1.0, help="Frequency scaling factor")
    parser.add_argument("--molden", type=Path, default=None, help="Write Molden file")
    parser.add_argument("--xyz", type=Path, default=None, help="Write XYZ file")


def _print_thermo(bundle) -> None:
    thermo = bundle.thermo
    print(
        f"\nThermochemistry (T={thermo.temperature} K, P={thermo.pressure} Pa, "
        f"sigma={thermo.symmetry_number}):"
    )
    print(f"  E (electronic, from .log) : {bundle.molecule.energy}")
    print(f"  ZPE                       : {thermo.zpe:14.8f} Hartree  "
          f"= {thermo.zpe*HARTREE_TO_KCALMOL:8.3f} kcal/mol")
    print(f"  U(T) - U(0)               : {thermo.thermal_correction_to_energy:14.8f} Hartree")
    print(f"  H(T) - U(0)               : {thermo.thermal_correction_to_enthalpy:14.8f} Hartree")
    print(f"  G(T) - U(0)               : {thermo.thermal_correction_to_gibbs:14.8f} Hartree")
    print(f"  S total                   : {thermo.s_total*HARTREE_TO_KJMOL*1000:10.3f} J/(mol K)")
    print(f"  Cv total                  : {thermo.cv_total*HARTREE_TO_KJMOL*1000:10.3f} J/(mol K)")
    print(f"  Cp total                  : {thermo.cp_total*HARTREE_TO_KJMOL*1000:10.3f} J/(mol K)")
    if bundle.molecule.energy is not None:
        E0 = bundle.molecule.energy
        print()
        print(f"  Total (E + ZPE)           : {E0 + thermo.zpe:14.8f} Hartree")
        print(f"  Total enthalpy            : {E0 + thermo.zpe + thermo.thermal_correction_to_enthalpy:14.8f} Hartree")
        print(f"  Total Gibbs               : {E0 + thermo.zpe + thermo.thermal_correction_to_gibbs:14.8f} Hartree")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nma-jax", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_min = sub.add_parser("minimum", help="Analyse a minimum")
    _add_common_args(p_min)

    p_ts = sub.add_parser("ts", help="Analyse a transition state")
    _add_common_args(p_ts)

    args = parser.parse_args(argv)

    func = analyse_minimum if args.cmd == "minimum" else analyse_transition_state
    bundle = func(
        args.fchk,
        log_path=args.log,
        temperature=args.temperature,
        pressure=args.pressure,
        symmetry_number=args.sigma,
        scale=args.scale,
        multiplicity=args.multiplicity,
    )

    print(f"Molecule: {bundle.molecule}")
    print(f"Imaginary frequencies: {bundle.n_imaginary}")
    print()
    print(format_frequencies(bundle.analysis))
    _print_thermo(bundle)

    if args.molden is not None:
        write_molden(args.molden, bundle.molecule, bundle.analysis)
        print(f"\nWrote {args.molden}")
    if args.xyz is not None:
        write_xyz(args.xyz, bundle.molecule, comment=str(args.fchk))
        print(f"Wrote {args.xyz}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
