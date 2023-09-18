"""
Author: Alex Hoffmann
Date: 3/11/2023
Description: Implementation of Wavelet Adaptive Interference Cancellation for Underdetermined Platforms
"""

import numpy as np
from wavelets import WaveletAnalysis
from scipy.ndimage import uniform_filter1d
import itertools
from scipy.signal import savgol_filter
scales = None

"Parameters"
fs = 1
dj = 1/12
detrend = True
uf = 400
denoise = False

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """

    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanWAICUP(B[:,axis,:])
        return(result)
    else:
        "B: (n_sensors, n_samples)"
        result = cleanWAICUP(B)
        return(result)
    
def cleanWAICUP(sensors):
    dt = 1/fs    

    "Detrend"
    if(detrend):
        trend = uniform_filter1d(sensors, size=uf)
        sensors -= trend
    
    if(sensors.shape[0] == 2): amb_mf = dual(sensors, dt, dj)
    else: amb_mf = multi(sensors, dt, dj)
    
    "Denoise"
    if(denoise):
        amb_mf = savgol_filter(amb_mf, 10, 2, mode='nearest')
    
    "Retrend"
    if(detrend):
        amb_mf += np.mean(trend, axis = 0)


    "Return Ambient Magnetic Field"
    return(amb_mf)
    

def dual(sig, dt, dj):
    "Create Wavelets"
    w1 = WaveletAnalysis(sig[0], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True)
    w2 = WaveletAnalysis(sig[1], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True)
    
    "Transform signals into wavelet domain"
    wn1 = w1.wavelet_transform.real
    wn2 = w2.wavelet_transform.real
    
    "Sheinker and Moldwin's Algorithm"
    dw = wn2-wn1
    wc1 = np.sum(dw*wn1, axis=1)
    wc2 = np.sum(dw*wn2, axis=1)
    k_hat_real = wc2/wc1
    w_clean_real = ((np.tile(k_hat_real,(wn1.shape[-1],1)).T*wn1)-wn2)/(np.tile(k_hat_real,(wn1.shape[-1],1)).T-1)
    
    "Record Scales"
    global scales
    scales = w1.scales
    
    "Transform to time domain"
    W_n = w_clean_real 
    Y_00 = w1.wavelet.time(0)
    s = w1.scales
    r_sum = np.sum(W_n.real.T / s ** .5, axis=-1).T
    amb_mf = r_sum * (dj * dt ** .5 / (w1.C_d * Y_00))
    amb_mf += w1.data.mean(axis=w1.axis, keepdims=True)
    return amb_mf

def multi(sig, dt, dj):
    "Find Combinations"
    pairs = list(itertools.combinations([i for i in range(sig.shape[0])], 2))
    waicup_level1 = np.zeros((len(pairs), sig.shape[-1]))
    
    w_obj = []
    for i in range(len(pairs)):
        waicup_level1[i] = dual(np.vstack((sig[pairs[i][0]], sig[pairs[i][1]])), dt, dj)
        w_obj.append(WaveletAnalysis(waicup_level1[i], dt=dt, frequency=True, dj = dj))
        
    w = [wav.wavelet_transform.real for wav in w_obj]
        
    "Iterate through Level 1 WAICUP"
    w = np.array(w)
    wn_clean = np.zeros(w[0].shape)

    "Iterate through every single datapoint"
    for row in range(wn_clean.shape[0]):
        for col in range(wn_clean.shape[1]):
            #print(row,col, wn.shape, wn_clean.shape)
            wn_clean[row,col] = w[np.argmin(np.abs(w[:,row,col])), row, col] #np.argmin(np.abs(wn[:,row,col]))
    
    "Reconstruct Ambient Magnetic Field Signal"
    W_n = wn_clean
    Y_00 = w_obj[0].wavelet.time(0)
    s = w_obj[0].scales
    r_sum = np.sum(W_n.real.T / s ** .5, axis=-1).T
    amb_mf = r_sum * (dj * dt ** .5 / (w_obj[0].C_d * Y_00))
    
    "Return Ambient Magnetic Field"
    return(amb_mf)

