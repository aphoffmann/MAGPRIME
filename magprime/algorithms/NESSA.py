"""
Author: Matt Finley, Alex Hoffmann
Last Update: 5/17/2023
Description: Ness M-SSA applies M-SSA to the high frequency components and
             Ness gradiometry to the low frequency components.

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data

Algorithm Parameters
----------
window_size : window size for MSSA
alpha : correlation threshold for identifying interference
variance_explained_threshold : variance explained threshold for MSSA
"""

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from pymssa import MSSA 

"Parameters"
uf = 400                                # Uniform Filter Size for detrending
detrend = True                          # Detrend the data

"Algorithm Parameters"
window_size = 400                       # Window size for MSSA
alpha = 0.05                            # Correlation threshold for identifying interference
variance_explained_threshold = 0.995    # Variance explained threshold for MSSA
aii = None                              # Coupling matrix between the sensors and sources for NESS

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    
    "Detrend and Clean Trend"
    if(detrend): 
        trend = uniform_filter1d(B, size=uf)
        B -= trend
        trend = cleanTrend(trend)

    "Apply M-SSA"
    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanMSSA(B[:,axis,:])
    else:
        result = cleanMSSA(B)

    "Restore Trend"
    if(detrend):
        result += trend

    return(result)
    

def cleanMSSA(sig):
    "Create MSSA Object and Fit"
    mssa = MSSA(n_components='variance_threshold',
               variance_explained_threshold=variance_explained_threshold,
               window_size=window_size,
               verbose=False)
    mssa.fit(sig.T)
    
    "Estimate Signal Interference"
    interference = sig[1] - sig[0]
    for i in range(1, sig.shape[0]-1):
        interference += (sig[i+1] - sig[i])
        
    "Take correlation of components and restore ambient magnetic field"
    components = mssa.components_[0].T
    amb_mf = np.zeros(interference.shape)
    interference_t = np.zeros(interference.shape)
    for c in range(components.shape[0]):
        corr = stats.pearsonr(interference, components[c])[0]
        if(np.abs(corr) > alpha): 
            interference_t += components[c]
        else:
            amb_mf += components[c]
            

    return(amb_mf)


def cleanTrend(B, triaxial = True):
    if(aii is None):
        raise("NESS.aii must be set before calling clean()")
    
    if(triaxial):
        result = np.multiply((B[0] - np.multiply(B[1], aii[:, np.newaxis])), (1/(1-aii))[:, np.newaxis])

    else:
        result = np.multiply((B[0] - np.multiply(B[1], aii)), (1/(1-aii)))
        
    return(result)