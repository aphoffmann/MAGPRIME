import numpy as np
from wavelets import WaveletAnalysis

"""
Author: Alex Hoffmann
Last Update: 5/17/2024
Description: Find Ness Coupling Coefficients through wavelet analysis 

Algorithm Parameters
----------
fs : sampling frequency
"""


"Algorithm Parameters"
fs = 1              # Sampling Frequency

def find_ness_coupling_coefficients(B):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    """

    # Take Wavelet Transform of the Magnetic Field Measurements
    # Filter out MSPs and ASSPs
    # Reconstruct Time Series
    # Fit Dipole to time series data

    # Return Ness Coupling Coefficients