# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  WAICUP.py                                                   ║
# ║  Package      :  magprime                                                    ║
# ║  Author       :  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-05-21                                                  ║
# ║  Last Updated :  2025-05-22                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  : Wavelet-Adaptive Interference Cancellation for               ║
# ║                 Underdetermined Platforms                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import numpy as np
from wavelets import WaveletAnalysis
from scipy.ndimage import uniform_filter1d
import itertools
from invertiblewavelets import Transform

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = True     # Detrend the data

"Algorithm Parameters"
fs = 1              # Sampling Frequency
dj = 1/12           # Wavelet Scale Spacing
scales = None       # Scales used in the wavelet transform
lowest_freq = None  # Lowest frequency in the wavelet transform
boom = None         # Trend to use during retrending process
filterbank = None   # Custom FilterBank Implimentation

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    if filterbank is None:
        if(triaxial):
            result = np.zeros((3, B.shape[-1]))
            for axis in range(3):
                result[axis] = cleanWAICUP(B[:,axis,:])
            return(result)
        
        else:
            "B: (n_sensors, n_samples)"
            result = cleanWAICUP(B)
            return(result)
        
    else:
        if(triaxial):
            result = np.zeros((3, B.shape[-1]))
            for axis in range(3):
                result[axis] = _clean_fb(B[:,axis,:], filterbank)
            return(result)
        
        else:
            result = _clean_fb(B, filterbank)
            return(result)
            

def cleanWAICUP(sensors):
    dt = 1/fs    

    "Detrend"
    if(detrend):
        trend = uniform_filter1d(sensors, size=uf)
        sensors = sensors - trend
    
    if(sensors.shape[0] == 2): result = dual(sensors, dt, dj)
    else: result = multi(sensors, dt, dj)
    
    "Retrend"
    if(detrend):
        if(boom is not None): result += trend[boom]
        else: result += np.mean(trend, axis = 0)

    return(result)
    

def dual(sig, dt, dj):
    "Create Wavelets"
    w1 = WaveletAnalysis(sig[0], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True)
    w2 = WaveletAnalysis(sig[1], dt=dt, frequency=True, dj = dj, unbias=False, mask_coi = True)

    if(lowest_freq is not None):
        w1.lowest_freq = lowest_freq
        w2.lowest_freq = lowest_freq
        
    
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
        wave_obj = WaveletAnalysis(waicup_level1[i], dt=dt, frequency=True, dj = dj)
        wave_obj.lowest_freq = lowest_freq
        w_obj.append(wave_obj)
        
    w = [wav.wavelet_transform.real for wav in w_obj]
        
    "Iterate through Level 1 WAICUP"
    w = np.array(w)
    abs_w = np.abs(w)
    indices = np.argmin(abs_w, axis=0)
    wn_clean = np.take_along_axis(w, indices[None, :, :], axis=0)[0]
    
    "Reconstruct Ambient Magnetic Field Signal"
    W_n = wn_clean
    Y_00 = w_obj[0].wavelet.time(0)
    s = w_obj[0].scales
    r_sum = np.sum(W_n.real.T / s ** .5, axis=-1).T
    amb_mf = r_sum * (dj * dt ** .5 / (w_obj[0].C_d * Y_00))
    
    "Return Ambient Magnetic Field"
    return(amb_mf)

def _clean_fb(B, fb):
    transform = Transform.from_filterbank(fb)
    pair_indices = list(itertools.combinations(range(B.shape[0]), 2))
    X_list = []

    # Step 1: WAICUP in the filterbank / wavelet domain
    for i, j in pair_indices:
        w1 = transform.forward(B[i], mode='full')   # → (J, T), complex
        w2 = transform.forward(B[j], mode='full')   # → (J, T), complex

        D   = w2 - w1                               # (J, T)
        C1  = np.sum(D * np.conj(w1), axis=1)       # (J,)
        C2  = np.sum(D * np.conj(w2), axis=1)       # (J,)
        K   = C2 / C1                               # (J,)

        # ambient-wavelet estimate X(s,τ)  eq(10)
        Xij = (K[:,None]*w1 - w2) / (K[:,None] - 1) # (J, T)
        X_list.append(Xij)

    # Step 2: pick the minimum‐magnitude X across all pairs
    X_stack = np.stack(X_list, axis=0)              # (n_pairs, J, T)
    abs_X   = np.abs(X_stack)                       # (n_pairs, J, T)
    winner  = np.argmin(abs_X, axis=0)              # (J, T)

    # fancy‐indexing to grab the winning X(s,τ) at each scale/time
    J, T = winner.shape
    jj, tt = np.indices((J, T))
    X_sel = X_stack[winner, jj, tt]                 # (J, T)

    # Step 3: one inverse transform back to the time series
    ambient = transform.inverse(X_sel, mode='full')  # (T,)

    return ambient