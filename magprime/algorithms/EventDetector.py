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
from scipy.fft import rfft

"General Parameters"
uf = 400                # Uniform Filter Size for detrending
detrend = False         # Detrend the data
use_fft = False         # Use FFT for feature extraction
save_segments = False   # Save the segment indices for visualization

"Variables for Anomaly Detection"
events = {}             # Dictionary to store detected events

# Detection of anomalous points
def anomaly_detection(input_data, sampling_rate_hz, window_length_sec, nu_value):
    """
    Detects anomalous segments in input_data using PCA and One-Class SVM.

    Parameters:
    - input_data: 1D numpy array of the signal.
    - sampling_rate_hz: Sampling rate in Hz.
    - window_length_sec: Length of each window in seconds.
    - nu_value: Parameter for One-Class SVM (nu).
    - use_fft: Boolean flag to apply rFFT on each segment.

    Returns:
    - anomaly_flag: 1D numpy array with anomaly flags for each data point.
    - ordered_anomalies: List of tuples with (start_index, stop_index) sorted by anomaly severity.
    """
    
    # Optional Detrending
    if detrend:
        input_data, _ = detrend(input_data, uf)

    window_length = int(window_length_sec * sampling_rate_hz)
    segments = [input_data[i:i+window_length] for i in range(0,int(window_length * np.floor(len(input_data)/window_length)),window_length)] 
    # Ensure that the input_data is divisible by window_length
    
    # Optional FFT Transformation
    if use_fft:
        # Apply rFFT to each segment and take the magnitude
        segments_fft = np.abs(rfft(segments, axis=1))
        # Optionally, you can normalize or scale the FFT features here
        features = segments_fft
    else:
        features = segments

    # Dimensionality Reduction via PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features)

    # Clustering via One-Class SVM
    ocsvm = svm.OneClassSVM(nu=nu_value, kernel='rbf', gamma='scale')
    ocsvm.fit(principal_components)
    


    # Predict anomalies: 1 for normal, -1 for anomalies
    y_pred = ocsvm.predict(principal_components)
    
    # Floor predictions at 0 for easy summation analysis
    y_pred = np.where(y_pred < 0, 0, 1)
   
    # Invert to time series
    anomaly_flag = np.zeros(len(input_data))
    for idx, pred in enumerate(y_pred):
        if pred == 0:
            start = idx * window_length
            stop = (idx + 1) * window_length
            anomaly_flag[start:stop] += 1

    if(save_segments):
        # Obtain anomaly scores (the lower, the more anomalous)
        anomaly_scores = ocsvm.decision_function(principal_components)
        
        # Create an ordered list of anomalies based on anomaly_scores
        # Since lower scores are more anomalous, we sort ascending
        ordered_indices = np.argsort(anomaly_scores)
        
        ordered_anomalies = []
        for idx in ordered_indices:
            score = anomaly_scores[idx]
            start = idx * window_length
            stop = (idx + 1) * window_length
            ordered_anomalies.append({
                'segment_index': idx,
                'start_index': start,
                'stop_index': stop,
                'anomaly_score': score
            })
        
        # Store the detected events by window length
        events[window_length_sec] = ordered_anomalies

    return anomaly_flag

# Detrending by uniform filter
def detrend(signal, filt_length):
    """
    Detrends the signal using a uniform filter.

    Parameters:
    - signal: 1D numpy array of the signal.
    - filt_length: Length of the uniform filter.

    Returns:
    - signal_detrended: Detrended signal.
    - trend: The trend component extracted from the signal.
    """
    trend = uniform_filter1d(signal, size=filt_length)
    signal_detrended = signal - trend
    return signal_detrended, trend
