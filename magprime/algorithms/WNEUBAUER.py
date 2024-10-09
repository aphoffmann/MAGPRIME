"""
Author: Alex Hoffmann
Last Update: 9/19/2023
Description: Todo

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data

Algorithm Parameters
----------
fs : sampling frequency
dj : wavelet scale spacing
scales : scales used in the wavelet transform (set by the algorithm)
"""

import numpy as np
from wavelets import WaveletAnalysis
from scipy.ndimage import uniform_filter1d
import itertools


"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = True     # Detrend the data

"Algorithm Parameters"
fs = 1              # Sampling Frequency
dj = 1/12           # Wavelet Scale Spacing
scales = None       # Scales used in the wavelet transform
lowest_freq = None  # Lowest frequency in the wavelet transform
boom = None         # Trend to use during retrending process
flip = False        # Flip the data before applying the algorithm

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    if(flip):
        B = np.flip(np.copy(B), axis = 0) 
    
    if(detrend):
        trend = uniform_filter1d(B, size=uf, axis = -1)
        B = B - trend
    
    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanWAICUP(B[:,axis,:])
    else:
        "B: (n_sensors, n_samples)"
        result = cleanWAICUP(B)

    "Retrend"
    if(detrend):
        if(boom is not None): result += trend[boom]
        else: result += np.mean(trend, axis = 0)
    
    return(result)

def cleanWAICUP(sensors):
    dt = 1/fs    

    n_sensors, n_samples = sensors.shape

    ## Take wavelet transform of each sensor
    wave_objs = []
    for i in range(n_sensors):
        w = WaveletAnalysis(sensors[i], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True)
        if lowest_freq is not None:
            w.lowest_freq = lowest_freq
        wave_objs.append(w.wavelet_transform.real)
    waves = np.array(wave_objs)

    ## Find gains for each wavelet scale for each sensor pair with sensor 0
    _, n_scales, _ = waves.shape
    gains = np.ones((n_sensors, n_sensors, n_scales))
    for i in range(1,n_sensors):
        dw = waves[i] - waves[0]
        wc1 = np.sum(dw*waves[0], axis=1)
        wc2 = np.sum(dw*waves[i], axis=1)
        k_hat = np.abs(wc2/wc1)
        gains[i, 1] = k_hat
        for j in range(2, n_sensors):
            gains[i, j] = k_hat * k_hat **((j-1) / 3)
        
    ## Calculate the ambient field for each scale
    w_clean = np.zeros((n_scales, n_samples))
    for scale in range(n_scales):
        det_mat6b = np.linalg.det(gains[:,:,scale])
        inv_mat6b = np.linalg.inv(gains[:,:,scale])
        adj_mat6b = det_mat6b * inv_mat6b.T
        C_col1 = adj_mat6b[:, 0] 
        B_amb = np.tensordot(C_col1, waves[:, scale, :], axes=([0], [0])) / det_mat6b  # Shape: (n_samples,)
        w_clean[scale, :] = B_amb # Shape: (n_scales, n_samples)

    ## Compare with boom
    if boom is not None:
        w = WaveletAnalysis(sensors[i], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True)
        if lowest_freq is not None:
            w.lowest_freq = lowest_freq
        w_boom = w.wavelet_transform.real
        wavs = np.array([w_clean, w_boom])
        abs_w = np.abs(wavs)
        indices = np.argmin(abs_w, axis=0)
        wn_clean = np.take_along_axis(wavs, indices[None, :, :], axis=0)[0]

    ## Reconstruct Ambient Magnetic Field Signal
    w1 = WaveletAnalysis(sensors[0], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True)
    if lowest_freq is not None:
        w.lowest_freq = lowest_freq
    W_n = w_clean 
    Y_00 = w1.wavelet.time(0)
    s = w1.scales
    r_sum = np.sum(W_n.real.T / s ** .5, axis=-1).T
    amb_mf = r_sum * (dj * dt ** .5 / (w1.C_d * Y_00))
    amb_mf += w1.data.mean(axis=w1.axis, keepdims=True)
    return amb_mf