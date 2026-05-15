"""Command-line interface for the ``nma`` package.

Two subcommands::

    nma analyze freq.log freq.fchk [--temperature 298.15] [--sigma 2]
    nma compare ref.fchk ref.log displaced.fchk displaced.log
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import (
    duschinsky,
    eckart_align,
    read_gaussian,
    thermochemistry,
    __version__,
)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scaling", type=float, default=1.0,
                   help="ZPE scaling factor (e.g. 0.965 for B3LYP/6-31G(d))")
    p.add_argument("--no-project-translation", dest="proj_t", action="store_false",
                   help="Do not project out translation (rare)")
    p.add_argument("--no-project-rotation", dest="proj_r", action="store_false",
                   help="Do not project out rotation (rare)")


def cmd_analyze(args: argparse.Namespace) -> int:
    mol = read_gaussian(args.log, args.fchk, zpe_scaling=args.scaling)
    print(f"Loaded {mol!r}")
    print(f"Centre of mass [Bohr]:  {mol.center_of_mass}")
    print(f"Linear?                 {mol.is_linear}")
    rc_cm = mol.rotational_constants_cm
    rc_ghz = mol.rotational_constants_ghz
    print(f"Rotational constants:   {rc_cm} cm^-1  ({rc_ghz} GHz)")

    modes = mol.normal_modes(
        project_translation=args.proj_t,
        project_rotation=args.proj_r,
    )
    print()
    print(modes.table())
    if mol.energy is not None and mol.zpe is not None:
        print(f"\nE_0     = {mol.energy: .8f}  Hartree")
        print(f"ZPE     = {mol.zpe: .8f}  Hartree  "
              f"(scaling x{args.scaling})")
        print(f"E + ZPE = {mol.energy + mol.zpe: .8f}  Hartree")

    if args.thermo:
        print()
        thermo = thermochemistry(
            modes,
            temperature=args.temperature,
            pressure=args.pressure,
            symmetry_number=args.sigma,
            spin_multiplicity=args.multiplicity,
        )
        print(thermo.summary())
        if mol.energy is not None:
            print(f"G = E + G_corr = {mol.energy + thermo.g_correction: .8f}  Hartree")
    if args.animate is not None:
        out = Path(args.animate)
        out.parent.mkdir(parents=True, exist_ok=True)
        modes.write_xyz_animation(out, mode_index=args.mode, amplitude=args.amplitude)
        print(f"Wrote animation of mode {args.mode} to {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ref = read_gaussian(args.ref_log, args.ref_fchk)
    disp = read_gaussian(args.disp_log, args.disp_fchk)
    eck = eckart_align(ref, disp)
    print(f"Mass-weighted RMSD after Eckart alignment: {eck.rmsd:.6f} Bohr")
    print(f"Eckart rotation matrix:\n{eck.rotation}")

    ref_modes = ref.normal_modes()
    disp_modes = disp.normal_modes()
    dusch = duschinsky(ref_modes, disp_modes)
    print(f"\nDuschinsky J shape: {dusch.J.shape}")
    print(f"Displacement K (mass-weighted, first 10 components):\n{dusch.K[:10]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct (but don't parse) the ``nma`` CLI parser.

    Split out from :func:`main` so it can be exercised by unit tests.
    """
    p = argparse.ArgumentParser(prog="nma", description=__doc__)
    p.add_argument("--version", action="version", version=f"nma {__version__}")
    sp = p.add_subparsers(dest="command", required=True)

    pa = sp.add_parser("analyze", help="Run normal-mode analysis on one geometry")
    pa.add_argument("log", help="Gaussian .log file")
    pa.add_argument("fchk", help="Gaussian .fchk file")
    pa.add_argument("--thermo", action="store_true",
                    help="Also compute RRHO thermochemistry")
    pa.add_argument("--temperature", type=float, default=298.15)
    pa.add_argument("--pressure", type=float, default=101_325.0)
    pa.add_argument("--sigma", type=int, default=1,
                    help="Rotational symmetry number")
    pa.add_argument("--multiplicity", type=int, default=1,
                    help="Spin multiplicity 2S+1")
    pa.add_argument("--animate", type=str, default=None,
                    help="Write XYZ animation to this path")
    pa.add_argument("--mode", type=int, default=0,
                    help="Mode index for --animate (default 0 = lowest)")
    pa.add_argument("--amplitude", type=float, default=0.5,
                    help="Displacement amplitude for --animate")
    _add_common_args(pa)
    pa.set_defaults(func=cmd_analyze, proj_t=True, proj_r=True)

    pc = sp.add_parser("compare", help="Eckart-align and Duschinsky-compare two geometries")
    pc.add_argument("ref_log")
    pc.add_argument("ref_fchk")
    pc.add_argument("disp_log")
    pc.add_argument("disp_fchk")
    pc.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    p = build_parser()
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
