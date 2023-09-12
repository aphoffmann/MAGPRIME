"""
Author: Alex Hoffmann
Date: 10/28/2022
Description: Implementation of gradiometry by Ness et al. (1971)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np

"Parameters"
r1 = None # Location of inboard sensor
r2 = None # Location of outboard sensor

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(r1 is None or r2 is None):
        raise("NESS.r1 and NESS.r2 must be set before calling clean()")

    if(triaxial):
        n_sensors = 2
        positions = np.array([r1,r2])
        couplings = np.zeros((n_sensors, n_sensors))
        for i in range(n_sensors):
            for j in range(n_sensors):
                couplings[i,j] = positions[i]**3 / positions[j]**3

        inv_couplings = np.linalg.pinv(couplings)  
        result = np.zeros(B.shape[1:])
        for i in range(B.shape[1]):
            result[i] = np.dot(inv_couplings[0,:], B[:,i,:])
        return(result)
    
    else:
        a = (r1/r2)**3
        result = (B[0] - a*B[1])/ (1-a)
        return(result)