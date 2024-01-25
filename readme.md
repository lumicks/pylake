<img src="https://media.githubusercontent.com/media/lumicks/pylake/main/docs/logo_light.png" alt="logo" width="489px"/>

[![DOI](https://zenodo.org/badge/133832492.svg)](https://zenodo.org/badge/latestdoi/133832492)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](license.md)
[![Build Status](https://github.com/lumicks/pylake/workflows/pytest/badge.svg?branch=main)](https://github.com/lumicks/pylake/actions/workflows/pylake_test.yml?query=branch%3Amain)
[![Documentation Status](https://readthedocs.org/projects/lumicks-pylake/badge/?version=latest)](https://lumicks-pylake.readthedocs.io/)

Pylake is a Python package for analyzing single-molecule optical tweezer data.

Its main features include:

- **Analyzing HDF5 and TIFF data obtained with Bluelake**: Pylake provides a convenient interface to read, analyze and plot data obtained with Bluelake.
- **Correlating data**: Provides functionality to correlate optical tweezer and imaging data.
- **Fitting force-extension curves**: Pylake provides tools to (globally) fit polymer models to force-extension data.
- **Force calibration routines**: The package contains algorithms to calibrate optical tweezer data using power spectral density estimation.
- **Kymotracker**: Tracks single molecules in a kymograph and computes binding rates and diffusion constants.

Please see the [documentation](https://lumicks-pylake.readthedocs.io/) for more information.

## Citing

Pylake is free to use under the conditions of the [Apache-2.0 open source license](license.md).

If you wish to publish results produced with this package, please mention the package name and cite the Zenodo DOI for this project:

[![DOI](https://zenodo.org/badge/133832492.svg)](https://zenodo.org/badge/latestdoi/133832492)

You'll find a *"Cite as"* section at the bottom right of the Zenodo page. You can select a citation
style from the dropdown menu or export the data in BibTeX and similar formats.
