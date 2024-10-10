"""
Author: Arie Sheinker, Alex Hoffmann
Last Update: 9/19/2023
Description: Todo

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data
"""
import numpy as np
from magprime.utility import calculate_coupling_coefficients
from wavelets import WaveletAnalysis

"Algorithm Parameters"
fs = 50
sspTol = 15
aii = None
weights = None
gain_method = 'sheinker' # 'sheinker' or 'ramen'
order = np.inf
flip = False        # Flip the data before applying the algorithm
treat_matrix = True

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """

    if(flip):
        B = np.flip(np.copy(B), axis = 0) 

    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanHOG(B[:,axis,:])

    else:
        "B: (n_sensors, n_samples)"
        result = cleanHOG(B)

    return(result)

def cleanHOG(B):
    n_sensors, n_samples = B.shape

    "Initialize coupling matrix K"
    global order
    order = min(order, B.shape[0])
    K = np.zeros((B.shape[0], order))
    K[:,0] = 1
    # set values above diagonals to 1: 
    for i in range(1,order): # Column
        K[i-1,i] = 1

    "Find first order coupling coefficients"
    for i in range(1,n_sensors):
        K[i,1] = findGain(B[[0,i]])


    "Calculate Gradients"
    gradients = [None, None]
    for i in range(1,n_sensors):
        a_ij = findGain(B[[i-1,i]])
        a_ij = findOptimalK(a_ij)
        B_sc = (B[i] - B[i-1]) / (a_ij - 1)
        gradients.append(B_sc)

    "Find higher order coupling coefficients"
    for i in range(2,order): # Column
        for j in range(i,n_sensors): # Row
            "Find Gain K[i,j]"
            K[j,i] = findGain(np.array([gradients[i], gradients[j+1]])) 

        "Recalculate Higher Order Gradients for next iteration"
        for j in range(i,n_sensors):
            a_ij = findGain(np.array([gradients[j], gradients[j+1]]))
            a_ij = findOptimalK(a_ij)
            G_sc = (gradients[j+1] - gradients[j]) / (a_ij - 1)
            gradients.append(G_sc)

    global aii;


    global weights;
    if(weights is None):
        weights = np.ones(n_sensors)
    W = np.diag(weights)

    if(treat_matrix):
        factors = np.geomspace(1, 100, 100)
        cond = np.linalg.cond(K.T @ W @ K)
        for factor in factors:
            K_temp = K.copy()
            for i in range(1, order):
                for j in range(i, n_sensors):
                    K_temp[j, i] *= factor

            if np.linalg.cond(K_temp.T @ W @ K_temp) < cond:
                #print("Condition number of K.T @ W @ K: ", np.linalg.cond(K_temp.T @ W @ K_temp), ", Factor: ", factor)
                K = K_temp
                cond = np.linalg.cond(K.T @ W @ K)

    aii = K
    det_mat6b = np.linalg.det(aii)
    inv_mat6b = np.linalg.inv(aii)
    adj_mat6b = det_mat6b * inv_mat6b.T
    C_col1 = adj_mat6b[:, 0] 

    B_amb = np.tensordot(C_col1, B, axes=([0], [0])) / det_mat6b
    #result = np.linalg.solve(K.T @ W @ K, K.T @ W @ B)
    #return(result[0])
    return(B_amb)

def findOptimalK(k):
    if not treat_matrix: return k

    K = np.array([[1,1],[1,k]])
    factors = np.geomspace(1, 100, 20)
    cond = np.linalg.cond(K.T @ K)
    for factor in factors:
        K_temp = K.copy()
        K_temp[1,1] *= factor
        if np.linalg.cond(K_temp.T @ K_temp) < cond:
            K = K_temp
            cond = np.linalg.cond(K.T @ K)
    return(K[1,1])


def findGain(B):
    global gain_method;
    if gain_method.lower() == "sheinker":
        return findGainSheinker(B)
    elif gain_method.lower() == "ramen":
        return findGainRamen(B)
    else:
        raise ValueError(f"Unknown gain method: {gain_method}")

def findGainSheinker(B):
    d = B[1]-B[0]
    c0 = np.abs(np.sum(d*B[0]))
    c1 = np.abs(np.sum(d*B[1]))
    k_hat = c1/c0
    return(k_hat)


def findGainRamen(B, fs=1, sspTol=15):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    fs : sampling frequency
    sspTol : cosine similarity threshold for identifying multi-source points (MSPs) and ambient single-source points (ASSPs)
    """
    
    # Take Wavelet Transform of the Magnetic Field Measurements
    w = WaveletAnalysis(B, dt=1/fs, frequency=True, dj = 1/12, unbias=False, mask_coi = True)

    # Filter out MSPs and ASSPs
    filtered_w = filter_wavelets(w.wavelet_transform, sspTol=sspTol) 
    
    # Reconstruct Time Series
    B_filtered = inverse_wavelet_transform(filtered_w, w)
    
    # Calculate Coupling Coefficients
    k_hat = np.nanmean(np.abs(B_filtered[1]) / np.abs(B_filtered[0]), axis=-1)

    return k_hat

def filter_wavelets(w, sspTol=15):
    """Filter out Multi Source Points (MSPs) and Ambient Single Source Points (ASSPs) from the wavelet transform of the magnetic field measurements"""

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

def inverse_wavelet_transform(filtered_w, w):
    """Apply Inverse Wavelet Transform to the Filtered data"""

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