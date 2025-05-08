'''
Title: LikelihoodRatio.py
Authors: Matthew G. Finley (NASA GSFC & UMD, College Park -- matthew.g.finley@nasa.gov)
Description: TODO
'''
import numpy as np
from scipy.stats import trim_mean
from scipy.ndimage import uniform_filter1d, median_filter

def likelihood_ratio(s, threshold=3):
    broadband_s = np.zeros(np.shape(s))
    narrowband_s = np.zeros(np.shape(s))
    for i in range(np.shape(s)[1]):
        raw_spectrum = s[:,i]
        broadband_spectrum = uniform_filter1d(raw_spectrum, 10)
        narrowband_spectrum = raw_spectrum - broadband_spectrum
        narrowband_spectrum = narrowband_spectrum - trim_mean(narrowband_spectrum, 0.10)

        broadband_s[:,i] = broadband_spectrum
        narrowband_s[:,i] = narrowband_spectrum

    SNR_ij = narrowband_s / broadband_s
    SNR_ij_p1 = SNR_ij + np.ones(np.shape(SNR_ij))
    Sb_ij = broadband_s

    test_statistic = (SNR_ij / SNR_ij_p1) * (s / Sb_ij)

    detected_pixels = (test_statistic > threshold).astype(np.uint8)

    median_filtered_detections = median_filter(detected_pixels, (1,3))

    return median_filtered_detections