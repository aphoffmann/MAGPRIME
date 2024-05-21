import numpy as np
from wavelets import WaveletAnalysis

"""
Author: Alex Hoffmann
Last Update: 5/17/2024
Description: Find Ness Coupling Coefficients through wavelet analysis 

Algorithm Parameters
----------
fs : sampling frequency
sspTol : cosine similarity threshold for identifying multi-source points
"""


"Algorithm Parameters"
fs = 1              # Sampling Frequency
sspTol = 15          # Cosine Similarity Threshold for Identifying Multi-Source Points
scales = None       # Scales used in the wavelet transform

def find_ness_coupling_coefficients(B):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    """
    
    # Take Wavelet Transform of the Magnetic Field Measurements
    w = WaveletAnalysis(B, dt=1/fs, frequency=True, dj = 1/12, unbias=False, mask_coi = True)
    scales = w.scales

    # Filter out MSPs and ASSPs
    filtered_w = filter_wavelets(w.wavelet_transform) # (n_scales, n_sensors, n_axes, n_samples)
    
    # Reconstruct Time Series
    B_filtered = inverse_wavelet_transform(filtered_w, w)
    
    # Calculate Coupling Coefficients
    alpha_couplings = np.nanmean((np.abs(B_filtered[0]) / np.abs(B_filtered[1])), axis=-1)

    return alpha_couplings

def inverse_wavelet_transform(filtered_w, w):
    # w: (n_scales, n_sensors, n_axes, n_samples)
    dt = 1/fs; dj = 1/12
    n_scales, n_sensors, n_axes, n_samples = filtered_w.shape
    result  = np.zeros((n_sensors, n_axes, n_samples))

    for i in range(n_axes):
        for j in range(n_sensors):
                W_n = filtered_w[:,j,i,:] 
                Y_00 = w.wavelet.time(0)
                r_sum = np.sum(W_n.real.T / scales ** .5, axis=-1).T
                amb_mf = r_sum * (dj * dt ** .5 / (w.C_d * Y_00))
                result[j,i,:] = amb_mf

    return result


# Define the filterSSP function for identifying multi-source points (MSPs)
def identify_MSP(B):
    """Identify Multi Source Points"""
    a = np.real(B)
    b = np.imag(B)
    a_dot_b = (a * b).sum(axis=0)
    norm_a = np.linalg.norm(a, axis=0)
    norm_a[norm_a == 0] = 1
    norm_b = np.linalg.norm(b, axis=0)
    norm_b[norm_b == 0] = 1
    cos_sim = np.abs(a_dot_b / (norm_a * norm_b))
    MSP_Bools = cos_sim < np.cos(np.deg2rad(sspTol))
    return MSP_Bools

# Define the identifyASSP function for identifying ambient single-source points (ambient SSPs)
def identify_ASSP(data):
    """Identify Ambient Single Source Points"""
    a = np.abs(data)
    b = np.ones(data.shape)
    a_dot_b = (a * b).sum(axis=0)
    norm_a = np.linalg.norm(a, axis=0)
    norm_a[norm_a == 0] = 1
    norm_b = np.linalg.norm(b, axis=0)
    norm_b[norm_b == 0] = 1
    cos_sim = np.abs(a_dot_b / (norm_a * norm_b))
    ASSP_Bools = cos_sim >= np.cos(np.deg2rad(sspTol))
    return ASSP_Bools


def filter_wavelets(w):
    n_scales, n_sensors, n_axes, n_samples = w.shape
    
    # Flatten scales
    w_flattened = w.transpose(1, 2, 0, 3).reshape(n_sensors, n_axes, n_scales * n_samples)
    
    for i in range(n_axes):        
        # Identify MSPs and zero them out
        MSP_Bools = identify_MSP(w_flattened[:, i, :])
        w_flattened[:, i, MSP_Bools] = 0
        
        # Identify ambient SSPs and zero them out
        ASSP_Bools = identify_ASSP(w_flattened[:, i, :])
        w_flattened[:, i, ASSP_Bools] = 0
    
    # Reshape back to original dimensions
    filtered_w = w_flattened.reshape(n_sensors, n_axes, n_scales, n_samples).transpose(2, 0, 1, 3)
    
    return filtered_w

