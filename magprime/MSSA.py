# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 14:55:02 2022

@author: Alex
"""

"""
Author: Alex Hoffmann
Date: 11/29/2022
Description: Noise Removal via Multichannel Singular Spectrum Analysis
"""
import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pymssa import MSSA 
import warnings
warnings.filterwarnings("ignore")

"Parameters"
window_size = 400 # Window size for MSSA
uf = 400 # Uniform Filter Size for detrending
alpha = 0.05 # Correlation threshold for identifying interference
detrend = True # Boolean for whether to detrend the signal

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
        sig -= trend
    
    "Create MSSA Object and Fit"
    mssa = MSSA(n_components='variance_threshold',
               variance_explained_threshold=0.995,
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