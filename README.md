# MAGPRIME: MAGnetic signal PRocessing, Interference Mitigation, and Enhancement

MAGPRIME (MAGnetic signal PRocessing, Interference Mitigation, and Enhancement) is an open-source Python library that provides a collection of noise removal algorithms tailored for space-based magnetic field measurements. The library aims to facilitate the development and testing of new noise removal techniques, as well as provide a comprehensive suite of existing algorithms for researchers and engineers working with magnetometer data.

## Features

- A variety of noise removal algorithms, including:
  - Wavelet-Adaptive Interference Cancellation for Underdetermined Platforms (Hoffmann and Moldwin, 2023)
  - Underdetermined Blind Source Separation (Hoffmann and Moldwin, 2022)
  - Adaptive Interference Cancellation for a pair of magnetometers (Sheinker and Moldwin, 2016)
  - Multivariate Singular Spectrum Analysis (Finley et al., 2023)
  - Principal Component Gradiometry (Constantinescu et al., 2020)
  - Independent Component Analysis (Imajo et al., 2021)
  - Traditional Gradiometry (Ness et al., 1971)
  - Frequency-based Gradiometry (Ream et al., 2021)
- Utility functions for data preprocessing and evaluation metrics
- Example scripts and datasets for demonstrating the usage of the library
- Benchmarks to compare interference removal performance

## Installation

MAGPRIME requires Python 3.9 or later. You can install the library using `pip`:

```bash
pip install magprime git+https://github.com/aphoffmann/MAGPRIME.git

```
Alternatively, you can clone the repository and install the library manually:

```
git clone https://github.com/your_username/MAGPRIME.git
cd MAGPRIME
python setup.py install
```
or 
```
git clone https://github.com/your_username/MAGPRIME.git
cd MAGPRIME
pip install .
```

## Usage
To use MAGPRIME, simply import the desired noise removal algorithm and apply it to your magnetic field data. For example, to use wavelet-based denoising


```
from magprime import ICA, MSSA, NESS, PiCoG, SHEINKER, REAM, UBSS, WAICUP

# Load the magnetometer data in the shape of b: (n_sensors, n_axes, n_samples)
b = load_data_function()

# Set any additional algorithm-specific arguments
UBSS.sigma = 10 # nT
UBSS.fs = 50 # Hz

# Clean the signals using the WAIC-UP algorithm
cleaned_signals = UBSS.clean(b) # returns (n_axes, n_samples)

# Perform further analysis or visualization with the cleaned_signals
# ...
```
For more detailed usage instructions and examples, please refer to the examples folder.

## Contributing
We welcome contributions to improve and expand the MAGPRIME library. If you would like to contribute, please follow these steps:

1. Fork the repository on GitHub.
2. Clone your forked repository to your local machine.
3. Create a new branch with a descriptive name for your feature or bugfix (git checkout -b my-feature-branch).
4. Make your changes, following the existing coding style and including comments and docstrings for any new functions or classes.
5. Test your changes to ensure that they work correctly and do not introduce new issues.
6. Commit your changes to your branch, using descriptive commit messages.
7. Push your changes to your forked repository on GitHub.
8. Open a pull request to the main repository, describing your changes and the motivation behind them.
Please ensure that your code follows the existing style guidelines and that you include unit tests for any new functionality, if applicable. Also, update the documentation as needed.


## Citation
If you use MAGPRIME in your research, please consider citing our paper:

[Your Name], [Co-author Names]. (Year). Advancing Space Science: A Critical Review of Magnetometer Noise Removal Techniques and the Development of an Open-Source Test-Bed. Journal Name, Volume(Issue), Pages. DOI: [DOI Number]
