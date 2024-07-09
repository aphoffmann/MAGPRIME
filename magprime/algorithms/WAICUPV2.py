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
gain_method = "sheinker" # 'sheinker' or 'ramen'
sspTol = 15         # Cosine similarity threshold for identifying multi-source points (MSPs) and ambient single-source points (ASSPs)
order = np.inf      # Order of the HOG algorithm
flip = False        # Flip the data before applying the algorithm


def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """

    if(flip):
        B = np.flip(np.copy(B), axis = 0) 

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
        trend = uniform_filter1d(sensors, size=uf, mode='constant')
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
    wn = np.array([wav.wavelet_transform for wav in w])

    scales = w[0].scales
    w_clean = np.zeros(wn.shape[1:])
    for s in range(w_clean.shape[0]):
        w_clean[s] = HOG(wn[:,s,:])

    W_n = w_clean.real 
    Y_00 = w[0].wavelet.time(0)
    r_sum = np.sum(W_n.real.T / scales ** .5, axis=-1).T
    amb_mf = r_sum * (dj * dt ** .5 / (w[0].C_d * Y_00.real))
    amb_mf += w[0].data.mean(axis=w[0].axis, keepdims=True)
    return amb_mf

def HOG(B):
    n_sensors, n_samples = B.shape

    "Initialize coupling matrix K"
    global order
    order = min(order, B.shape[0])
    K = np.zeros((B.shape[0], order))
    K[:,0] = 1
    # set values above diagonals to 1: 
    for i in range(1,order): # Column
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
    for i in range(2,order): # Column
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
    cond = np.linalg.cond(K)
    for factor in factors:
        K_temp = K.copy()
        for i in range(1, order):
            for j in range(i, n_sensors):
                K_temp[j, i] *= factor**j

        if np.linalg.cond(K_temp.T @ W @ K_temp) < cond:
            K = K_temp
            cond = np.linalg.cond(K)
    result = np.linalg.solve(K.T @ W @ K, K.T @ W @ B)
    return(result[0])


def findGain(B):
    if gain_method.lower() == "sheinker":
        return findGainShienker(B.real)
    elif gain_method.lower() == "ramen":
        return findGainRAMEN(B)
    else:
        raise ValueError("Invalid gain method")

def findGainShienker(B):
    d = B[1]-B[0]
    c0 = np.sum(d*B[0])
    c1 = np.sum(d*B[1])
    k_hat = c1/c0
    return(k_hat)

def findGainRAMEN(B):
        B_filtered = np.copy(B)
        
        # Identify MSPs and zero them out
        MSP_Bools = identify_MSP(B_filtered, sspTol=sspTol)
        B_filtered[:, MSP_Bools] = 0
        
        # Identify ambient SSPs and zero them out
        ASSP_Bools = identify_ASSP(B_filtered, sspTol=sspTol)
        B_filtered[:, ASSP_Bools] = 0

        k_hat = np.nanmean(np.abs(B_filtered[1]) / np.abs(B_filtered[0]), axis=-1)
        if(np.isnan(k_hat) or np.isinf(k_hat)):
            k_hat = 1.1
        return(k_hat)



def identify_MSP(B, sspTol=15):
    """Identify Multi Source Points"""
    a = np.real(B)
    b = np.imag(B)
    a_dot_b = (a * b).sum(axis=0)
    norm_a = np.linalg.norm(a, axis=0)
    norm_a[norm_a == 0] = 1
    norm_b = np.linalg.norm(b, axis=0)
    norm_b[norm_b == 0] = 1
    cos_sim = np.abs(a_dot_b / (norm_a * norm_b))
    MSP_Bools = cos_sim < np.cos(np.deg2rad(sspTol))
    return MSP_Bools

def identify_ASSP(data, sspTol=15):
    """Identify Ambient Single Source Points"""
    a = np.abs(data)
    b = np.ones(data.shape)
    a_dot_b = (a * b).sum(axis=0)
    norm_a = np.linalg.norm(a, axis=0)
    norm_a[norm_a == 0] = 1
    norm_b = np.linalg.norm(b, axis=0)
    norm_b[norm_b == 0] = 1
    cos_sim = np.abs(a_dot_b / (norm_a * norm_b))
    ASSP_Bools = cos_sim >= np.cos(np.deg2rad(sspTol))
    return ASSP_Bools