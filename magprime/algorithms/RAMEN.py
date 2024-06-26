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
weights = None      # Weights for the Least-Squares Fit

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(len(B.shape) > 2 and not triaxial):
        raise Exception("Exception: Triaxial Selected but B has more than 2 dimensions")

    global aii
    if(aii is None):
        aii = calculate_coupling_coefficients(B, fs=fs, sspTol=sspTol, triaxial=triaxial)

    result = cleanNess(B, triaxial)

    return(result)


def cleanNess(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    global weights
    if(weights is None):
        weights = np.ones(B.shape[0])

    if(weights.shape[0] != aii.shape[0]):
        raise Exception("Exception: Weights must have the same number of elements as the number of sensors")

    W = np.diag(weights)
    result = np.zeros((2,3, B.shape[-1]))
    if(triaxial):
        for axis in range(3):
            # Get the mixing matrix for the current axis
            A = aii[axis]
            
            # Solve for the ambient magnetic field signal
            result[:, axis, :] = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ np.flip(np.copy(B[:, axis, :]), axis = 0)

    else:
        A = aii
        # Solve for the ambient magnetic field signal
        result = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ np.flip(np.copy(B), axis = 0)
        
    return(result[0])
    


def calculate_coupling_coefficients(B, fs=1, sspTol=15, triaxial=True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    fs : sampling frequency
    sspTol : cosine similarity threshold for identifying multi-source points (MSPs) and ambient single-source points (ASSPs)
    """
    
    # Take Wavelet Transform of the Magnetic Field Measurements
    w = WaveletAnalysis(B, dt=1/fs, frequency=True, dj = 1/12, unbias=False, mask_coi = True)

    # Filter out MSPs and ASSPs
    filtered_w = filter_wavelets(w.wavelet_transform, sspTol=sspTol, triaxial=triaxial) 
    
    # Reconstruct Time Series
    B_filtered = inverse_wavelet_transform(filtered_w, w, triaxial=triaxial)
    
    # Calculate Coupling Coefficients
    alpha_couplings = calculate_mixing_matrix(B_filtered, triaxial=triaxial)

    return alpha_couplings

def calculate_mixing_matrix(B_filtered, triaxial):
    """Calculate the mixing matrix for the magnetic field measurements"""
    n_sensors = B_filtered.shape[0]

    if(triaxial):
        mixing_matrices = []
        for axis in range(3):
            # Initialize the mixing matrix with ones in the first column
            mixing_matrix = np.zeros((n_sensors, 2))
            mixing_matrix[:, 0] = 1
            
            # Calculate alpha values for the upper triangular part of the matrix for this axis
            for i in range(n_sensors):
                alpha_ij = np.nanmean(np.abs(B_filtered[0, axis]) / np.abs(B_filtered[i, axis]), axis=-1)
                mixing_matrix[i,1] = alpha_ij
            mixing_matrices.append((mixing_matrix))
        return np.array(mixing_matrices)
    
    else:
        mixing_matrix = np.zeros((n_sensors, 2))
        mixing_matrix[:, 0] = 1
        
        for i in range(n_sensors):
            alpha_ij = np.nanmean(np.abs(B_filtered[0]) / np.abs(B_filtered[i]), axis=-1)
            mixing_matrix[i,1] = alpha_ij
        return mixing_matrix
    
def filter_wavelets(w, sspTol=15, triaxial=True):
    """Filter out Multi Source Points (MSPs) and Ambient Single Source Points (ASSPs) from the wavelet transform of the magnetic field measurements"""
    if(triaxial):
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
    else:
        n_scales, n_sensors, n_samples = w.shape
        
        # Flatten scales
        w_flattened = w.transpose(1, 0, 2).reshape(n_sensors, n_scales * n_samples)
        
        # Identify MSPs and zero them out
        MSP_Bools = identify_MSP(w_flattened, sspTol=sspTol)
        w_flattened[:, MSP_Bools] = 0
        
        # Identify ambient SSPs and zero them out
        ASSP_Bools = identify_ASSP(w_flattened, sspTol=sspTol)
        w_flattened[:, ASSP_Bools] = 0
        
        # Reshape back to original dimensions
        filtered_w = w_flattened.reshape(n_sensors, n_scales, n_samples).transpose(1, 0, 2)
        
        return filtered_w

def inverse_wavelet_transform(filtered_w, w, triaxial=True):
    """Apply Inverse Wavelet Transform to the Filtered data"""
    if(triaxial):
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
    else:
        _, n_sensors, n_samples = filtered_w.shape
        result  = np.zeros((n_sensors, n_samples))

        for j in range(n_sensors):
            W_n = filtered_w[:,j,:] 
            Y_00 = w.wavelet.time(0)
            r_sum = np.sum(W_n.real.T / w.scales ** .5, axis=-1).T
            amb_mf = r_sum * (w.dj * w.dt ** .5 / (w.C_d * Y_00))
            result[j,:] = np.real(amb_mf)

        return result

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