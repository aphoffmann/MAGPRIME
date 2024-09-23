"""
Author: Alex Hoffmann
Last Update: 9/19/2023
Description: 
             
General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data

Algorithm Parameters
----------
delta_B : threshold for the change in the differenced field envelope (nT)
n : number of time steps for the change in the envelope
p : percentile threshold for identifying spectral peaks (0-100)
"""

import pandas as pd
import numpy as np
from scipy.signal import windows
from scipy.ndimage import uniform_filter1d

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

"Algorithm Parameters"
delta_B = None      # Threshold for the change in the differenced field envelope (nT)
n = 10              # number of time steps for the change in the envelope
p = 98              # percentile threshold for identifying spectral peaks (0-100)

def clean(B, triaxial = True):
    """
    Perform magnetic gradiometry using frequency-domain filtering
    Input:
        B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    Output:
        result: reconstructed ambient field without the spacecraft-generated fields (axes, n_samples)
    """
    # Check if delta_B has been set.
    if delta_B is None:
        raise ValueError("REAM.delta_B must be set before calling clean()")

    if(detrend):
        trend = uniform_filter1d(B, size=uf, axis = -1)
        B = B - trend

    if(triaxial):
        result = np.zeros(B.shape[1:])
        for axis in range(3):
            result[axis] = gradiometry_filter(B[0,axis,:], B[1,axis,:])
    else:
        result = gradiometry_filter(B[0], B[1])
        
    if(detrend):
        result += np.mean(trend, axis=0)
    
    return(result)

def gradiometry_filter(B1, B2):
    """
    Perform magnetic gradiometry using frequency-domain filtering
    Input:
        B1: magnetic field measurements from the inboard sensor
        B2: magnetic field measurements from the outboard sensor
    Output:
        B_amb: reconstructed ambient field without the spacecraft-generated fields
    """
    
    # Calculate the differenced field
    B_diff = B2 - B1
    
    # Calculate the rolling maximum and minimum of the differenced field
    delta_B_series = pd.Series(B_diff)
    B_max = delta_B_series.rolling(1920, center=True, min_periods=1).max()
    B_min = delta_B_series.rolling(1920, center=True, min_periods=1).min()
    
    # Identify the intervals where the differenced field changes significantly
    dB = np.abs(B_max - B_min) / n # change in the envelope
    mask = (dB > delta_B).to_numpy() # boolean mask for the threshold condition
    intervals = [] # list of intervals
    start = None # start index of an interval
    for i in range(len(mask)):
        if mask[i] and start is None: # start of an interval
            start = i
        elif (not mask[i] or i == len(mask)-1) and start is not None: # end of an interval
            end = i
            if end - start >= n: # check if the interval is long enough
                intervals.append((start, end))
            start = None # reset start index
    
    # Initialize the output array
    B_amb = np.copy(B1)
    
    # Loop over each interval
    for start, end in intervals:
        # Mirror the data at the edges of the interval
        B1_mir = np.concatenate((B1[end:start:-1], B1[start:end], B1[end:start:-1]))
        B2_mir = np.concatenate((B2[end:start:-1], B2[start:end], B2[end:start:-1]))
        B_diff_mir = np.concatenate((B_diff[end:start:-1], B_diff[start:end], B_diff[end:start:-1]))

        # Window the data
        win = windows.kaiser(len(B1_mir), beta=10)
        B1_win = B1_mir * win # windowed inboard sensor data
        B2_win = B2_mir * win # windowed outboard sensor data
        B_diff_win = B_diff_mir * win # windowed differenced data     
        
        # Perform the FFT
        F1 = np.fft.fft(B1_win)
        F2 = np.fft.fft(B2_win)
        F_diff = np.fft.fft(B_diff_win)

        # Compute the power spectra of the differenced field
        P_diff = np.abs(F_diff)**2

        # Identify the spectral peaks in the differenced field spectrum using a percentile threshold
        threshold = np.percentile(P_diff, 98)
        peak_indices = np.where(P_diff > threshold)[0]

        # Suppress the peaks in the inboard and outboard sensor spectra
        F1_suppr = F1.copy()
        F2_suppr = F2.copy()

        for peak in peak_indices:
            F1_suppr[peak] *= 0.01
            F2_suppr[peak] *= 0.01

        B1_suppr = np.fft.ifft(F1_suppr) / win
        B2_suppr = np.fft.ifft(F2_suppr) / win

        # Remove the mirrored sections and divide by window
        half_window_len = len(win) // 3
        B1_suppr = B1_suppr[half_window_len:-half_window_len] 
        B2_suppr = B2_suppr[half_window_len:-half_window_len] 

        # Reconstruct the ambient field at each sensor
        B_interval = (B1_suppr + B2_suppr) / 2

        # Correct the bias of the reconstructed field B_interval based on the start and end points outside of the interval
        B1_bias = np.mean(B1[start:end])
        B2_bias = np.mean(B2[start:end])
        B_interval_bias = np.mean(B_interval)
        B_interval = B_interval + (B1_bias + B2_bias) / 2 - B_interval_bias
        B_amb[start:end] = B_interval

    return(B_amb)