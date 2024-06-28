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
weights = None      # Weights for the Least-Squares Fit

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
        sensors = sensors - trend
    
    result = dual(sensors, dt, dj)

    "Retrend"
    if(detrend):
        new_trend = HOG(trend)
        result += new_trend
    return(result)
    

def dual(sig, dt, dj):
    "Create Wavelets"
    w = np.array([WaveletAnalysis(sig[i], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True) for i in range(sig.shape[0])])
    wn = np.array([wav.wavelet_transform.real for wav in w])

    scales = w[0].scales
    w_clean_real = np.zeros(wn.shape[1:])
    for s in range(w_clean_real.shape[0]):
        w_clean_real[s] = HOG(wn[:,s,:])

    W_n = w_clean_real 
    Y_00 = w[0].wavelet.time(0)
    r_sum = np.sum(W_n.real.T / scales ** .5, axis=-1).T
    amb_mf = r_sum * (dj * dt ** .5 / (w[0].C_d * Y_00.real))
    amb_mf += w[0].data.mean(axis=w[0].axis, keepdims=True)
    return amb_mf

def HOG(B):
    n_sensors, n_samples = B.shape

    "Initialize coupling matrix K"
    K = np.zeros((B.shape[0], B.shape[0]))
    K[:,0] = 1
    # set values above diagonals to 1: 
    for i in range(1,n_sensors): # Column
        K[i-1,i] = 1

    "Find first order coupling coefficients"
    for i in range(1,n_sensors):
        K[i,1] = findGain(B[[0,i]])


    "Calculate Gradients"
    gradients = [None, None]
    for i in range(1,n_sensors):
        a_ij = findGain(B[[i-1,i]])
        B_sc = (B[i] - B[i-1]) / (a_ij - 1)
        gradients.append(B_sc)

    "Find higher order coupling coefficients"
    for i in range(2,n_sensors): # Column
        for j in range(i,n_sensors): # Row
            "Find Gain K[i,j]"
            K[j,i] = findGain(np.array([gradients[i],gradients[j+1]]))

        "Recalculate Higher Order Gradients for next iteration"
        for j in range(i,n_sensors):
            a_ij = findGain(np.array([gradients[j], gradients[j+1]]))
            G_sc = (gradients[j+1] - gradients[j]) / (a_ij - 1)
            gradients.append(G_sc)

    global weights
    if(weights is None):
        weights = np.ones(n_sensors)
    W = np.diag(weights)

    factors = np.geomspace(1, 100, 100)
    cond = np.linalg.cond(K.T @ W @ K)
    for factor in factors:
        K_temp = K.copy()
        for i in range(1, n_sensors):
            for j in range(i, n_sensors):
                K_temp[j, i] *= factor

        if np.linalg.cond(K_temp.T @ W @ K_temp) < cond:
            K = K_temp
            cond = np.linalg.cond(K.T @ W @ K)

    aii = K
    result = np.linalg.solve(K.T @ W @ K, K.T @ W @ B)
    return(result[0])



def findGain(B):
    d = B[1]-B[0]
    c0 = np.sum(d*B[0])
    c1 = np.sum(d*B[1])
    k_hat = c1/c0
    return(k_hat)

