# Normal Mode Analysis

A python program to perform harmonic vibrational analysis for a gas phase isolated molecule. Currently, the quantum chemsitry output is taken from the Gaussian program. However, the program is general and can be interfaced with any electronic structure package.

## What you need to run the program

- python3
- numpy

## Gaussian output files

The log file and the *formatted* checkpoint files from a Gaussian calculation are needed to run this program. The path can be defined in ```run_normal_mode_analysis.py```. The path for these files can be specificed in the input.

## Running the program

Once the input parameters have been defined, the program can be invoked as:

```
python run_normal_mode_analysis.py
```

If executed successfully, the program will produce a plain text output file named as ```transform_cartesian_normal``` which not only have the normal modes and frequencies, but also has the coordinate transformation matrices for converting cartesian coordinates to normal coordinates and vice-versa. This is helpful for many applications.

## Running the tests

The unit tests can be run by invoking the following command in the top-level directory:

```
python -m unittest discover -v
```

## Documentation

https://mowgliamu.github.io/NormalModeAnalysis/
