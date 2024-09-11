"""
Author: Arie Sheinker, Alex Hoffmann
Last Update: 9/19/2023
Description: Todo

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data
"""
import numpy as np
from scipy.ndimage import uniform_filter1d


"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """

    if(detrend):
        trend = uniform_filter1d(B, size=uf, axis = -1)
        B = B - trend

    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanSheinker(B[:,axis,:])

    else:
        "B: (n_sensors, n_samples)"
        result = cleanSheinker(B)
    
    if(detrend):
        result += np.mean(trend, axis=0)

    return(result)

def cleanSheinker(sig):
    d = sig[1]-sig[0]
    c0 = np.sum(d*sig[0])
    c1 = np.sum(d*sig[1])
    k_hat = c1/c0
    clean_sig = (k_hat*sig[0]-sig[1]) / (k_hat - 1)
    return(clean_sig)