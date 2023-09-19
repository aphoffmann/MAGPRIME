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
aii : Coupling matrix between the sensors and sources for NESS

"""

import numpy as np

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

"Algorithm Parameters"
aii = None # Coupling matrix between the sensors and sources for NESS

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(aii is None):
        raise("NESS.aii must be set before calling clean()")
    
    if(detrend):
        trend = uniform_filter1d(B, size=uf, axis = -1)
        B -= trend

    result = cleanNess(B, triaxial)

    if(detrend):
        result += np.mean(trend, axis=0)

    return(result)


def cleanNess(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(aii is None):
        raise("NESS.aii must be set before calling clean()")
    
    if(triaxial):
        result = np.multiply((B[0] - np.multiply(B[1], aii[:, np.newaxis])), (1/(1-aii))[:, np.newaxis])

    else:
        result = np.multiply((B[0] - np.multiply(B[1], aii)), (1/(1-aii)))
        
    return(result)
    