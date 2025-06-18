'''
Title: 
    AnomalyTracker.py
Authors: 
    Matthew G. Finley (NASA GSFC & UMD, College Park -- matthew.g.finley@nasa.gov)
Description: 
    RUDE (Rapid Unsupervised Detection of Events)-based method of spectral track detection.
    Portions of this methodology originally described in 'Generalized Time Series Analysis
    for In Situ Spacecraft Observations: Anomaly Detection and Data Prioritization Using
    Principal Components Analysis and Unsupervised Clustering' by Finley et al., Earth
    and Space Science (2024).
'''
from magprime.algorithms.anomaly.EventDetector import anomaly_detection
from magprime.utility import load_crm_data
import numpy as np

def anomaly_tracker(s, window_length=10, nu=0.1):
    power_at_each_time = []
    for col in range(np.shape(s)[1]):
        power_at_each_time.append(np.log(s[:,col]))

    flags_at_each_time = []
    for time_step in power_at_each_time:
        time_step_flags = anomaly_detection(time_step, 1, window_length, nu)
        flags_at_each_time.append(time_step_flags)

    output_image = np.zeros(np.shape(s))
    for idx, observation in enumerate(flags_at_each_time):
        output_image[:,idx] = observation

    return output_image