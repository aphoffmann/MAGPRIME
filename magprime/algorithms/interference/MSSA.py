# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  MSSA.py                                                     ║
# ║  Package      :  magprime                                                    ║
# ║  Author       :  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-05-21                                                  ║
# ║  Last Updated :  2025-05-22                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  : Multivariate Singular Spectrum Analysis for                  ║
# ║                 interference mitigation                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
"""
Author: Matt Finley, Alex Hoffmann
Last Update: 9/19/2023
Description: Todo

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
detrend = False                         # Detrend the data

"Algorithm Parameters"
window_size = 400                       # Window size for MSSA
alpha = 0.05                            # Correlation threshold for identifying interference
variance_explained_threshold = 0.995    # Variance explained threshold for MSSA

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanMSSA(B[:,axis,:])
        return(result)
    else:
        result = cleanMSSA(B)
        return(result)
    

def cleanMSSA(sig):
    "Detrend"
    if(detrend): 
        trend = uniform_filter1d(sig, size=uf)
        sig = sig - trend
    
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
            
    "Retrend"
    if(detrend):
        amb_mf += np.mean(trend, axis = 0)
    
    return(amb_mf)