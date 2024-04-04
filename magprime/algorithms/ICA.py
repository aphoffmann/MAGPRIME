"""
Author: Shun Imajo, Alex Hoffmann
Last Update: 9/19/2023
Description: This file implements a noise removal method using independent component
             analysis (ICA) for magnetic field data from multiple magnetometers. 
             The method follows the approach of Imajo et al. (2021). The method 
             separates the noise components from the signal components based on 
             their statistical independence and non-Gaussianity. The natural magnetic 
             field components are then identified by their similarity in the mixing
             matrix (ica.mixing_).
             
General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data
"""

from sklearn.decomposition import FastICA
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    if(triaxial):
        result = cleanTriAxis(B)
    else:
        result = cleanAxis(B)

    return(result)
        
    

def cleanAxis(B):
    n_components = B.shape[0]
    ica = FastICA(n_components=n_components, whiten=False, max_iter=20000, tol = 1e-8)
    
    "Remove Trend"
    if(detrend): 
        trend = uniform_filter1d(B, size=uf)
        B -= trend
    
    "Apply ICA"
    X = B.T # (n_samples, n_features)
    S_ = ica.fit_transform(X)  
    S_ = S_.T
    
    "Find Ambient IC through lowest correlation with the difference"
    diff = np.zeros(B.shape[-1])
    for i in range(B.shape[0]-1):
        diff += (B[i+1] - B[i])

    r = np.zeros(B.shape[0])
    for i in range(B.shape[0]):
        r[i] = np.abs(stats.pearsonr(diff, S_[i])[0])
        
    "Reapply Trend"
    recovered_signal = S_[np.argmin(r)]
    
    if(detrend):
        recovered_signal += np.mean(trend, axis = 0)

    return(recovered_signal)

def cleanTriAxis(B):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    """
    n_sensors, n_axes, n_samples = B.shape
    B_transposed = B.transpose(1, 0, 2) # (axes, n_sensors, n_samples) 
    sig = B_transposed.reshape(n_sensors * n_axes, n_samples)

    n_components = sig.shape[0]
    ica = FastICA(n_components=n_components, whiten="unit-variance", max_iter=1000)
    
    "Remove Trend"
    if(detrend): 
        trend = uniform_filter1d(sig, size=uf)
        sig -= trend
    
    "Apply ICA"
    X = sig.T # (n_samples, n_features)
    S_ = ica.fit_transform(X)  
    S_ = S_.T
    
    "Find Natural IC through lowest correlation with the difference"
    expected_mixing = np.ones(n_sensors)
    gain = np.ones(n_axes)
    args = []
    step = n_sensors
    for axis in range(n_axes):
        axis_mixing = ica.mixing_[axis*step:(axis+1)*step]
        cosine_similarities = [np.dot(col, expected_mixing) / (np.linalg.norm(col) * np.linalg.norm(expected_mixing)) for col in axis_mixing.T]
    
        args.append(np.argmax(np.abs(cosine_similarities)))
        gain[axis] = np.mean(axis_mixing.T[np.argmax(np.abs(cosine_similarities))])

    gain = list(gain)

    # Todo find gain through averaging the amb field vector using args
        
    "Select IC's with lowest correlation with the difference and reapply trend"
    recovered_signal = np.zeros((3, sig.shape[-1]))

    if(detrend):
        recovered_signal[0] = S_[args[0]]*gain[0] + np.mean(trend[:step], axis = 0)
        recovered_signal[1] = S_[args[1]]*gain[1] + np.mean(trend[step:2*step], axis = 0) 
        recovered_signal[2] = S_[args[2]]*gain[2] + np.mean(trend[2*step:3*step], axis = 0) 
    else:
        recovered_signal[0] = S_[args[0]]*gain[0]
        recovered_signal[1] = S_[args[1]]*gain[1] 
        recovered_signal[2] = S_[args[2]]*gain[2]


    return(recovered_signal)