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

def clean(B, triaxis = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(r1 is None or r2 is None):
        raise("NESS.r1 and NESS.r2 must be set before calling clean()")
    
    a = (r1/r2)**3

    if(triaxis):
        result = np.zeros(B.shape[1:])
        for axis in range(3):
            result[axis] = (B[0,axis,:] - a*B[1,axis,:])/ (1-a)
        return(result)
    else:
        result = (B[0] - a*B[1])/ (1-a)
        return(result)