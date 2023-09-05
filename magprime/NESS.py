"""
Author: Alex Hoffmann
Date: 10/28/2022
Description: Implementation of gradiometry by Ness et al. (1971)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
def clean(sig, r1 = 1, r2 = 2):
    '''
    

    Parameters
    ----------
    sig : TYPE
        DESCRIPTION.
    r1 : TYPE, optional
        DESCRIPTION. The default is 1.
    r2 : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    '''
    a = (r1/r2)**3
    recovered_signal = (sig[1] - a*sig[0])/(1-a)
    return(recovered_signal)