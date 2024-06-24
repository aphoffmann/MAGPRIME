"""
Author: Alex Hoffmann
Last Update: 6/24/2024
Description: Reduction Algorithm for Magnetometer Environmental Noise

Algorithm Parameters
----------
aii : Coupling matrix between the sensors and sources for NESS
fs : sampling frequency
sspTol : cosine similarity threshold for identifying multi-source points (MSPs) and ambient single-source points (ASSPs)

"""

import numpy as np
from wavelets import WaveletAnalysis

"Algorithm Parameters"
aii = None          # Coupling matrix between the sensors and sources for NESS
fs = 1              # Sampling Frequency
sspTol = 15         # Cosine similarity threshold for identifying multi-source points (MSPs) and ambient single-source points (ASSPs)

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(aii is None):
        aii = calculate_coupling_coefficients(B, fs=fs, sspTol=sspTol, triaxial=triaxial)

    result = cleanNess(B, triaxial)

    return(result)


def cleanNess(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(aii is None):
        raise("NESS.aii must be set before calling clean()")
    
    if(triaxial):
        result = np.multiply((B[0] - np.multiply(B[1], aii[:, np.newaxis])), (1/(1-aii))[:, np.newaxis])

    else:
        result = np.multiply((B[0] - np.multiply(B[1], aii)), (1/(1-aii)))
        
    return(result)
    


def calculate_coupling_coefficients(B, fs=1, sspTol=15, triaxial=True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    fs : sampling frequency
    sspTol : cosine similarity threshold for identifying multi-source points (MSPs) and ambient single-source points (ASSPs)
    """
    
    # Take Wavelet Transform of the Magnetic Field Measurements
    w = WaveletAnalysis(B, dt=1/fs, frequency=True, dj = 1/12, unbias=False, mask_coi = True)

    # Filter out MSPs and ASSPs
    filtered_w = filter_wavelets(w.wavelet_transform, sspTol=sspTol) # (n_scales, n_sensors, n_axes, n_samples)
    
    # Reconstruct Time Series
    B_filtered = inverse_wavelet_transform(filtered_w, w)
    
    # Calculate Coupling Coefficients
    alpha_couplings = np.nanmean((np.abs(B_filtered[0]) / np.abs(B_filtered[1])), axis=-1)

    return alpha_couplings

def inverse_wavelet_transform(filtered_w, w):
    # w: (n_scales, n_sensors, n_axes, n_samples)
    _, n_sensors, n_axes, n_samples = filtered_w.shape
    result  = np.zeros((n_sensors, n_axes, n_samples))

    for i in range(n_axes):
        for j in range(n_sensors):
                W_n = filtered_w[:,j,i,:] 
                Y_00 = w.wavelet.time(0)
                r_sum = np.sum(W_n.real.T / w.scales ** .5, axis=-1).T
                amb_mf = r_sum * (w.dj * w.dt ** .5 / (w.C_d * Y_00))
                result[j,i,:] = np.real(amb_mf)

    return result


# Define the filterSSP function for identifying multi-source points (MSPs)
def identify_MSP(B, sspTol=15):
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
def identify_ASSP(data, sspTol=15):
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


def filter_wavelets(w, sspTol=15):
    n_scales, n_sensors, n_axes, n_samples = w.shape
    
    # Flatten scales
    w_flattened = w.transpose(1, 2, 0, 3).reshape(n_sensors, n_axes, n_scales * n_samples)
    
    for i in range(n_axes):        
        # Identify MSPs and zero them out
        MSP_Bools = identify_MSP(w_flattened[:, i, :], sspTol=sspTol)
        w_flattened[:, i, MSP_Bools] = 0
        
        # Identify ambient SSPs and zero them out
        ASSP_Bools = identify_ASSP(w_flattened[:, i, :], sspTol=sspTol)
        w_flattened[:, i, ASSP_Bools] = 0
    
    # Reshape back to original dimensions
    filtered_w = w_flattened.reshape(n_sensors, n_axes, n_scales, n_samples).transpose(2, 0, 1, 3)
    
    return filtered_w

