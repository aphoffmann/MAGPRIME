"""
Author: Alex Hoffmann
Date: 10/24/2022
Description: Dual Magnetometer Interference Cancellation by Sheinker and Moldwin (2016)
"""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def clean(sig):
    """
    

    Parameters
    ----------
    sig : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    d = sig[1]-sig[0]
    c0 = np.sum(d*sig[0])
    c1 = np.sum(d*sig[1])
    k_hat = c1/c0
    clean_sig = (k_hat*sig[0]-sig[1]) / (k_hat - 1)
    return(clean_sig)