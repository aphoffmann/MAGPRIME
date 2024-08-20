""" 
Title: sample_detection_utilities.py

Date/Version: 17 April 2024 (v0.1)

Author: Matthew G. Finley (NASA GSFC/University of Maryland)

Contact: matthew.g.finley@nasa.gov

Description: Utility file to detrend data and test for anomalous samples. This utility file is used in 'sample_detection.py' and accompanies the manuscript
    'Generalized Time-Series Analysis for In-Situ Spacecraft Observations: Anomaly Detection and Data Prioritization using Principal Components Analysis and Unsupervised Clustering' 
    by M.G. Finley, M. Martinez-Ledesma, W.R. Paterson, M.R. Argall, D.M. Miles, J.C. Dorelli, and E. Zesta.

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data
    
"""

from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm
from scipy.ndimage import uniform_filter1d

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

# Detection of anomalous points
def anomaly_detection(input_data, sampling_rate_hz, window_length_sec, nu_value):
    # Dimensionality reduction via PCA

    if detrend:
        input_data, _ = detrend(input_data, uf)

    window_length = int(window_length_sec * sampling_rate_hz)
    intervals = [input_data[i:i+window_length] for i in range(0,int(window_length * np.floor(len(input_data)/window_length)),window_length)] # The floor math ensures that there are no 'short' intervals < interval_length

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(intervals)

    # Clustering via one-class SVM
   

    ocsvm = svm.OneClassSVM(nu=nu_value, kernel='rbf', gamma='scale')
    ocsvm.fit(principal_components)
    y_pred = ocsvm.predict(principal_components)

    y_pred[y_pred < 0] = 0 # Floor at zero for easy summation analysis
   
    # Invert to time series
    anomaly_flag = np.zeros(len(input_data))
    window_start = 0
    window_stop = window_length
    for i in y_pred:
        if i != 1:
            anomaly_flag[window_start:window_stop] += 1
        window_start = window_stop
        window_stop = window_stop + window_length

    return anomaly_flag

# Detrending by uniform filter
def detrend(signal, filt_length):

    trend = uniform_filter1d(signal, size=filt_length)
    signal_detrended = signal - trend
    return signal_detrended, trend