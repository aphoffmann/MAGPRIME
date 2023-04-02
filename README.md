# MAGPRIME: Magnetometer Noise Removal Library for Space-Based Applications

MAGPRIME (MAGnetic PRocessing, Interference Mitigation, and Enhancement) is an open-source Python library that provides a collection of noise removal algorithms tailored for space-based magnetic field measurements. The library aims to facilitate the development and testing of new noise removal techniques, as well as provide a comprehensive suite of existing algorithms for researchers and engineers working with magnetometer data.

## Features

- A variety of noise removal algorithms, including:
  - Wavelet-Adaptive Interference Cancellation for Underdetermined Platforms (Hoffmann and Moldwin, 2023)
  - Underdetermined Blind Source Separation (Hoffmann and Moldwin, 2022)
  - Adaptive Interference Cancellation for a pair of magnetometers (Sheinker and Moldwin, 2016)
  - Multivariate Singular Spectrum Analysis (Finley et al., 2023)
  - Principal component analysis (PCA)
  - Independent component analysis (Imajo et al., 2021)
  - Gradiometry Algorithms
    - Ness
    - Ream
- Utility functions for data preprocessing and evaluation metrics
- Example scripts and datasets for demonstrating the usage of the library

## Installation

MAGPRIME requires Python 3.6 or later. You can install the library using `pip`:

```bash
pip install magprime
