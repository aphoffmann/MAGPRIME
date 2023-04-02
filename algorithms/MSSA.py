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

def clean(sig, detrend = True, window_size = 400, uf = 200, alpha = 0.05):
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