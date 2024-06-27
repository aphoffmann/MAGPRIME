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

"Algorithm Parameters"
fs = 50
sspTol = 15
aii = None

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """

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
    K = np.zeros((B.shape[0], B.shape[0]))
    K[:,0] = 1
    # set values above diagonals to 1: 
    for i in range(1,n_sensors): # Column
        K[i-1,i] = 1

    "Find first order coupling coefficients"
    for i in range(1,n_sensors):
        K[i,1] = findGain(B[[i,0]])


    "Calculate Gradients"
    gradients = [None, None]
    for i in range(1,n_sensors):
        a_ij = findGain(B[[i,i-1]])
        B_sc = (B[i] - B[i-1]) / (a_ij - 1)
        gradients.append(B_sc)

    "Find higher order coupling coefficients"
    for i in range(2,n_sensors): # Column
        for j in range(i,n_sensors): # Row
            "Find Gain K[i,j]"
            K[j,i] = findGain(np.array([ gradients[j+1],gradients[i]]))

        "Recalculate Higher Order Gradients for next iteration"
        for j in range(i,n_sensors):
            a_ij = findGain(np.array([ gradients[j+1], gradients[j]]))
            G_sc = (gradients[j+1] - gradients[j]) / (a_ij - 1)
            gradients.append(G_sc)

    global aii;
    aii = K

    W = np.diag(np.geomspace(10, 0.1, n_sensors))
    result = np.linalg.inv(K.T @ W @ K) @ K.T @ W @ np.flip(B, axis=0)
    return(result[0])

def findGain(B):
    d = B[1]-B[0]
    c0 = np.sum(d*B[0])
    c1 = np.sum(d*B[1])
    k_hat = c1/c0
    return(k_hat)