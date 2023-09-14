"""
Author: Alex Hoffmann
Date: 10/28/2022
Description: Implementation of gradiometry by Ness et al. (1971)
"""
import numpy as np

"Parameters"
aii = None # Coupling matrix between the sensors and sources for NESS

def clean(B, triaxial = True):
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
    