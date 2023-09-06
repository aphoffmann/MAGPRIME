"""
Author: Alex Hoffmann
Date: 10/24/2022
Description: Noise Removal via independent component analysis. This algorithm
             is based on Imajo et al. (2021), but is applied to only one axis.
             The noise IC's are identified through their correlation with the
             difference between the magnetometers.
"""
from sklearn.decomposition import FastICA
import numpy as np
from scipy import stats
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

import warnings
warnings.filterwarnings("ignore")

"Parameters"
fs = 1 # Sampling Frequency
uf = 400 # Uniform Filter Size for detrending
detrend = True

def clean(B, triaxis = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """

    if(triaxis):
        recovered_signal = cleanTriAxis(B)
    else:
        recovered_signal = cleanAxis(B)
    return(recovered_signal)
        
    

def cleanAxis(sig):
    n_components = sig.shape[0]
    ica = FastICA(n_components=n_components, whiten=False, max_iter=20000, tol = 1e-8)
    
    "Remove Trend"
    if(detrend): 
        trend = uniform_filter1d(sig, size=uf)
        sig -= trend
    
    "Apply ICA"
    X = sig.T # (n_samples, n_features)
    S_ = ica.fit_transform(X)  
    S_ = S_.T
    
    "Find Ambient IC through lowest correlation with the difference"
    diff = np.zeros(sig.shape[-1])
    for i in range(sig.shape[0]-1):
        diff += (sig[i+1] - sig[i])

    r = np.zeros(sig.shape[0])
    for i in range(sig.shape[0]):
        r[i] = np.abs(stats.pearsonr(diff, S_[i])[0])
        
    "Reapply Trend"
    recovered_signal = S_[np.argmin(r)] #+ np.mean(filtered, axis = 0) 
    
    if(detrend):
        recovered_signal += np.mean(trend, axis = 0)

    return(recovered_signal)



def cleanTriAxis(sig):
    "TODO: Center and Whiten"
    n_components = sig.shape[0]
    ica = FastICA(n_components=n_components, whiten="unit-variance", max_iter=1000)
    
    "Remove Trend"
    filtered = uniform_filter1d(sig, size=uf)
    sig -= filtered

    
    "Apply ICA"
    X = sig.T # (n_samples, n_features)
    S_ = ica.fit_transform(X)  
    S_ = S_.T
    
    
    "Find Natural IC"
    step = ica.mixing_.shape[0]//3
    diffs = np.abs([ica.mixing_[i] - ica.mixing_[i-1] for i in range(step-1,ica.mixing_.shape[0],step)])
    args = np.argmin(diffs, axis = 1)
    gain = np.abs([(ica.mixing_[i] + ica.mixing_[i-1])/2 for i in range(step-1,ica.mixing_.shape[0],step)])
    
    "Restore Ambient ICs"
    recovered_signal = np.zeros((3, sig.shape[-1]))
    recovered_signal[0] = S_[args[0]]*gain[0][args[0]] + np.mean(filtered[:step], axis = 0)
    recovered_signal[1] = S_[args[1]]*gain[1][args[1]] + np.mean(filtered[step:2*step], axis = 0) 
    recovered_signal[2] = S_[args[1]]*gain[2][args[2]] + np.mean(filtered[2*step:3*step], axis = 0) 
        
    return(recovered_signal)