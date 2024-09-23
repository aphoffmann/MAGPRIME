"""
Author: Matt Finley, Alex Hoffmann
Last Update: 9/19/2023
Description: Todo

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data

Algorithm Parameters
----------
window_size : window size for MSSA
alpha : correlation threshold for identifying interference
variance_explained_threshold : variance explained threshold for MSSA
"""

import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
from pymssa import MSSA 
from joblib import Parallel, delayed

"Parameters"
uf = 400                                # Uniform Filter Size for detrending
detrend = False                         # Detrend the data

"Algorithm Parameters"
window_size = 400                       # Window size for MSSA
alpha = 0.05                            # Correlation threshold for identifying interference
variance_explained_threshold = 0.995    # Variance explained threshold for MSSA
boom = 0

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """
    if(triaxial):
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanMSSA(B[:,axis,:])
        return(result)
    else:
        result = cleanMSSA(B)
        return(result)
    

def cleanMSSA(sig):
    "Detrend"
    if(detrend): 
        trend = uniform_filter1d(sig, size=uf)
        sig = sig - trend
    
    "Create MSSA Object and Fit"
    mssa = MSSA(n_components='variance_threshold',
               variance_explained_threshold=variance_explained_threshold,
               window_size=window_size,
               verbose=False)
    mssa.fit(sig.T)
    
    "Estimate Signal Interference"
    interference = sig[1] - sig[0]
    for i in range(1, sig.shape[0]-1):
        interference += (sig[i+1] - sig[i])
        
    "Take correlation of components and restore ambient magnetic field"
    components = mssa.components_[0].T
    amb_mf = np.zeros(interference.shape)
    interference_t = np.zeros(interference.shape)
    for c in range(components.shape[0]):
        corr = stats.pearsonr(interference, components[c])[0]
        if(np.abs(corr) > alpha): 
            interference_t += components[c]
        else:
            amb_mf += components[c]
            
    "Retrend"
    if(detrend):
        amb_mf += np.mean(trend, axis = 0)
    
    return(amb_mf)


def cleanMSSA2(B):
    # Parameters
    N = B.shape[-1]  # Signal length
    L = window_size  # Window length for MSSA
    K = N - L + 1  # Second dimension of trajectory matrix

    if K <= 0:
        raise ValueError("Window size L is too large for the signal length N.")

    # Generate trajectory matrices
    grand_traj = np.array([B[:, i:i + K] for i in range(L)])
    grand_traj = np.concatenate(grand_traj, axis=0)

    # Generate covariance matrix
    C = np.dot(grand_traj, grand_traj.T) / K

    # Calculate eigenvalues and eigenvectors of C
    lambda_, V = np.linalg.eig(C)
    ind = np.argsort(lambda_)[::-1]  # Decreasing order
    lambda_ = lambda_[ind]
    V = V[:, ind]

    # Project trajectory matrix onto eigenvectors
    P = np.dot(V.T, grand_traj)

    # Generate reconstructions
    RC = np.zeros((B.shape[0], 2 * L, N))

    def compute_rc(l):
        rc_l = np.zeros((B.shape[0], N))
        for sensor in range(B.shape[0]):
            buf = np.dot(P[l, :].reshape(-1, 1), V[sensor * L:(sensor + 1) * L, l].reshape(1, -1))
            buf = buf[::-1, :]
            for n in range(N):
                diag_elements = np.diag(buf, n - K)
                if diag_elements.size > 0:
                    rc_l[sensor, n] = np.mean(diag_elements)
                else:
                    rc_l[sensor, n] = 0  # Handle empty diagonals
        return rc_l

    RC = np.array(Parallel(n_jobs=-1)(delayed(compute_rc)(l) for l in range(2 * L)))
    RC = RC.transpose(1, 0, 2)  # Reorder to shape (B.shape[0], 2 * L, N)

    # Find ambient magnetic field through correlation
    interference = B[-1] - B[0]
    amb_mf = np.zeros(interference.shape)
    interference_t = np.zeros(interference.shape)
    for c in range(RC.shape[1]):
        corr = stats.pearsonr(interference[L:-L], RC[boom, c,L:-L])[0]
        if np.abs(corr) > alpha:
            interference_t += RC[boom, c]
        else:
            amb_mf += RC[boom, c]

    return amb_mf
