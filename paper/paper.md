---
title: "`nma_jax`: differentiable normal-mode analysis in JAX"
tags:
  - Python
  - JAX
  - quantum chemistry
  - vibrational spectroscopy
  - normal-mode analysis
  - Franck-Condon
  - Duschinsky
authors:
  - name: Prateek Goel
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 15 May 2026
bibliography: paper.bib
---

# Summary

`nma_jax` is a Python package for harmonic vibrational analysis of molecules,
built on JAX [@jax2018github]. Given a Hessian matrix from an electronic
structure calculation (Gaussian [@gaussian16] or ORCA [@neese2020]), it
returns vibrational frequencies, mass-weighted and Cartesian normal-mode
columns, rotational constants, rigid-rotor-harmonic-oscillator
thermochemistry at arbitrary temperature and pressure, mass-weighted
Eckart alignment of two geometries [@kabsch1976; @umeyama1991], and
Duschinsky rotation matrices for vibronic-spectroscopy applications
[@duschinsky1937]. Every numerical routine is a pure JAX function:
`jit`-compilable, `vmap`-vectorisable, and differentiable through
`jax.grad` and `jax.jacrev`. This makes `nma_jax` directly composable with
neural-network potentials and any other gradient-based pipeline.

# Statement of need

Normal-mode analysis is a routine post-processing step in computational
chemistry, but the tooling landscape is fragmented. Quantum-chemistry
packages such as Gaussian, ORCA, and Psi4 [@psi4] each contain their own
internal implementations, and general-purpose simulation libraries such
as ASE [@ase2017] provide vibrational-analysis utilities. Each of these
solves a slightly different problem in a slightly different way, and
none expose the analysis as differentiable operations that can be
embedded inside a larger machine-learning or optimisation workflow.

In parallel, a growing community of researchers training neural-network
interatomic potentials [@behler2007; @schutt2018; @batzner2022] now
routinely needs vibrational properties at scale: frequencies for
training-set augmentation, Hessians for sensitivity analysis, harmonic
free energies for thermodynamic integration. Embedding a non-differentiable
Fortran NMA routine inside a JAX or PyTorch training loop is awkward;
re-implementing the analysis from scratch is error-prone (the sign
convention on the moment-of-inertia tensor is a famous trap, as is the
distinction between the body-frame and space-frame projection of
infinitesimal rotations [@miller1980; @page1988]).

`nma_jax` fills this gap by providing a clean, modern, end-to-end-differentiable
implementation of the standard mass-weighted Hessian analysis. The
public API is built around frozen dataclasses (`Molecule`, `NormalModes`,
`Thermo`, `EckartResult`, `DuschinskyResult`) and a handful of free
functions; the underlying numerics live in a small `core` module of
pure JAX routines. The package also implements two features that are
rarely available out-of-the-box in other packages: mass-weighted Eckart
alignment via the Kabsch–Umeyama singular-value decomposition with a
$\det = +1$ correction, and computation of the Duschinsky rotation matrix
$J$ and displacement vector $K$ between two electronic states. These are
exactly the inputs needed to compute multidimensional Franck–Condon
factors [@doktorov1977] for vibrationally resolved electronic spectra.

# Functionality

The principal entry point is the `Molecule` class, which holds atomic
numbers, masses, Cartesian coordinates (Bohr), and optional Hessian,
gradient, energy, and zero-point energy. The associated method
`Molecule.normal_modes()` returns a `NormalModes` object containing the
$3N-6$ (or $3N-5$ for linear molecules) vibrational frequencies in
cm$^{-1}$, both Cartesian and mass-weighted normal-mode columns, and
the per-mode gradient if a gradient was supplied. Imaginary frequencies
are reported as negative real numbers throughout, eliminating the
sentinel-value bookkeeping common in legacy codes.

The `thermochemistry()` function implements rigid-rotor-harmonic-oscillator
statistical thermodynamics at arbitrary $T$ and $P$, with the
Sackur–Tetrode translational contribution, asymmetric-top and linear
rotational branches, harmonic vibrational contributions (with explicit
imaginary-mode handling for transition states), and an electronic
contribution from the spin multiplicity.

The `eckart_align()` function solves

$$\Phi(U) = \sum_a m_a \left\Vert R^{(B)}_a - U\, R^{(A)}_a \right\Vert^2$$

for the proper rotation $U$ minimising the mass-weighted RMSD between
two geometries, returning both the rotation matrix and the achieved
RMSD. The companion `duschinsky()` function takes two `NormalModes`
objects and returns the rotation matrix $J$ relating their mass-weighted
normal coordinates and the displacement vector $K$ projected onto the
displaced state's normal modes. An accompanying Jupyter notebook
demonstrates the full pipeline from two electronic states to a
Gaussian-broadened Franck–Condon spectrum on a water example.

The package ships with parsers for Gaussian (`.log` + `.fchk`) and
ORCA (`.hess` + `.out`); a command-line interface (`nma analyze`,
`nma compare`) provides quick inspection without writing a script;
and an XYZ-trajectory exporter writes one period of any normal mode
for visualisation in VMD [@vmd] or Molden [@molden]. A test suite of
30 unit tests covers sign conventions, the trans/rot projection,
linear and non-linear molecules, Eckart alignment, Duschinsky
round-trips, transformation-matrix inversion, thermochemistry, and
both QC-package parsers.

# Acknowledgements

This package began as a rewrite of unpublished doctoral code. The
author thanks the open-source quantum-chemistry community whose
documentation, test cases, and decades of accumulated convention-wisdom
made the modernisation tractable.

# References
