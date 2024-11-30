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
from scipy.optimize import minimize_scalar
from numpy.linalg import pinv

"Algorithm Parameters"
aii = None          # Coupling matrix between the sensors and sources for NESS
fs = 1              # Sampling Frequency
sspTol = 15         # Cosine similarity threshold for identifying multi-source points (MSPs) and ambient single-source points (ASSPs)
weights = None      # Weights for the Least-Squares Fit
neubauer = False    # Boolean for whether to use Neubauer's method for calculating the coupling coefficients

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


def cleanNess(B, triaxial=True, neubauer=False):
    """
    Clean the magnetic field measurements by solving for the ambient magnetic field signal S0
    using Cramer's Rule, without applying a weighting matrix.

    Parameters:
    - B: numpy array of shape (n_sensors, n_axes, n_samples)
         Magnetic field measurements from the sensor array.
    - triaxial: bool, default=True
         Indicates whether to use triaxial or uniaxial ICA.
    - neubauer: bool, default=False
         Indicates whether to apply Neubauer's method.

    Returns:
    - S0: numpy array
         The solved ambient magnetic field signal S0.
         - If triaxial=True, shape is (n_axes, n_samples).
         - If triaxial=False, shape is (n_samples,).
    """

    global aii  # Assuming 'aii' (mixing matrices) is defined globally elsewhere in your code

    # Initialize the result array based on 'neubauer' and 'triaxial' parameters
    if neubauer:
        # If Neubauer's method requires handling additional components (e.g., a^3 and a^4),
        # initialize the result array accordingly. Adjust the shape as needed based on your specific requirements.
        # Here, we'll assume 'neubauer=True' maintains the original shape of B.
        result_S0 = np.zeros(B.shape)
    else:
        if triaxial:
            result_S0 = np.zeros((B.shape[1], B.shape[2]))  # (n_axes, n_samples)
        else:
            result_S0 = np.zeros(B.shape[2])  # (n_samples,)

    if triaxial:
        for axis in range(B.shape[1]):  # Loop over each axis (e.g., x, y, z)
            # Get the mixing matrix for the current axis
            A = aii[axis]  # Shape: (n_sensors, m_sources)

            # Compute determinant of KTK
            det_A = np.linalg.det(A)
            if det_A == 0:
                raise ValueError(f"Singular matrix for axis {axis}. Cannot solve for S0.")

            # Compute adjugate of KTK
            adj_A = det_A * np.linalg.inv(A).T  # adj(KTK) = det(KTK) * inv(KTK).T

            # Extract the first column of adjugate matrix (corresponding to S0)
            C_col1 = adj_A[:, 0]  # Shape: (m_sources,)

            # Prepare B for this axis and all time samples
            B_sorted = np.flip(B[:, axis, :], axis=0)  # Shape: (n_sensors, n_samples)

            # Compute the dot product of C_col1 and B_sorted across sensors
            # Using tensordot to handle multiple samples efficiently
            S0 = np.tensordot(C_col1, B_sorted, axes=([0], [0])) / det_A  # Shape: (n_samples,)

            if neubauer:
                # Assign S0 accordingly if Neubauer's method is applied
                result_S0[:, axis, :] = S0  # Shape assumed to be (n_samples,)
            else:
                # Assign S0 to the result array
                result_S0[axis, :] = S0  # Populate for current axis
    else:
        # Process uniaxial data
        A = aii  # Shape: (n_sensors, m_sources)

        # Compute determinant of KTK
        det_A = np.linalg.det(A)
        if det_A == 0:
            raise ValueError("Singular matrix. Cannot solve for S0.")

        # Compute adjugate of KTK
        adj_A = det_A * np.linalg.inv(A).T  # adj(KTK) = det(KTK) * inv(KTK).T

        # Extract the first column of adjugate matrix (corresponding to S0)
        C_col1 = adj_A[:, 0]  # Shape: (m_sources,)

        # Prepare B for all time samples
        B_sorted = np.flip(B, axis=0)  # Shape: (n_sensors, n_samples)

        # Compute S0
        S0 = np.tensordot(C_col1, B_sorted, axes=([0], [0])) / det_A  # Shape: (n_samples,)

        if neubauer:
            # Assign S0 accordingly if Neubauer's method is applied
            result_S0 = S0  # Shape assumed to be (n_samples,)
        else:
            # Assign S0 to the result array
            result_S0 = S0  # Shape: (n_samples,)

    return result_S0
    


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
    """Calculate the mixing matrix for the magnetic field measurements."""
    n_sensors = B_filtered.shape[0]
    #global neubauer
    print("Neubauer", neubauer)
    if triaxial:
        mixing_matrices = []
        for axis in range(3):
            if neubauer:
                
                # Extend the mixing matrix to shape (n_sensors, 3)
                mixing_matrix = np.ones((n_sensors, 3))

                # Calculate alpha_ij and decompose into a^3 and a^4 for this axis
                for i in range(1, n_sensors):
                    # Calculate alpha_ij
                    alpha_ij = np.nanmean(np.abs(B_filtered[0, axis]) / np.abs(B_filtered[i, axis]))
                    
                    # Define the function to minimize
                    def f(a):
                        return (a**3 - alpha_ij)**2
                    
                    # Perform the minimization with bounds to ensure positive 'a'
                    result = minimize_scalar(f)
                    a = result.x
                    
                    # Compute a_i^3 and a_i^4
                    a_i3 = a**3
                    a_i4 = a**4
                    
                    # Assign to the mixing matrix
                    mixing_matrix[i, 1] = a_i3
                    mixing_matrix[i, 2] = a_i4
                mixing_matrices.append(mixing_matrix)
            else:
                # Original mixing matrix code without Neubauer's method
                mixing_matrix = np.ones((n_sensors, 2))
                for i in range(1, n_sensors):
                    alpha_ij = np.nanmean(np.abs(B_filtered[0, axis]) / np.abs(B_filtered[i, axis]))
                    mixing_matrix[i, 1] = alpha_ij
                mixing_matrices.append(mixing_matrix)
        return np.array(mixing_matrices)
    else:
        if neubauer:
            # Extend the mixing matrix to shape (n_sensors, 3)
            mixing_matrix = np.ones((n_sensors, 3))

            # Calculate alpha_ij and decompose into a^3 and a^4
            for i in range(1, n_sensors):
                # Calculate alpha_ij
                alpha_ij = np.nanmean(np.abs(B_filtered[0]) / np.abs(B_filtered[i]))
                
                # Define the function to minimize
                def f(a):
                    return (a**3 + a**4 - alpha_ij)**2
                
                # Perform the minimization with bounds to ensure positive 'a'
                result = minimize_scalar(f, bounds=(0, None), method='bounded')
                a = result.x
                
                # Compute a_i^3 and a_i^4
                a_i3 = a**3
                a_i4 = a**4
                
                # Assign to the mixing matrix
                mixing_matrix[i, 1] = a_i3
                mixing_matrix[i, 2] = a_i4
            return mixing_matrix
        else:
            # Original mixing matrix code without Neubauer's method
            mixing_matrix = np.ones((n_sensors, 2))
            for i in range(1, n_sensors):
                alpha_ij = np.nanmean(np.abs(B_filtered[0]) / np.abs(B_filtered[i]))
                mixing_matrix[i, 1] = alpha_ij
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