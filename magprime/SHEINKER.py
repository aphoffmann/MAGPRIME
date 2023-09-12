"""
Author: Alex Hoffmann
Date: 10/24/2022
Description: Dual Magnetometer Interference Cancellation by Sheinker and Moldwin (2016)
"""
import numpy as np

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanSheinker(B[:,axis,:])
        return(result)
    else:
        "B: (n_sensors, n_samples)"
        result = cleanSheinker(B)
        return(result)

def cleanSheinker(sig):
    d = sig[1]-sig[0]
    c0 = np.sum(d*sig[0])
    c1 = np.sum(d*sig[1])
    k_hat = c1/c0
    clean_sig = (k_hat*sig[0]-sig[1]) / (k_hat - 1)
    return(clean_sig)