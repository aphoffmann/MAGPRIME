# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  PiCoG.py                                                    ║
# ║  Package      :  magprime                                                    ║
# ║  Author       :  Dr. Ovidiu Dragos Constantinescu                            ║
# ║                  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  Institut für Geophysik und extraterrestrische Physik        ║
# ║                  NASA Goddard Space Flight Center                            ║
# ║  Created      :  2025-05-21                                                  ║
# ║  Last Updated :  2025-07-19                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  : Principal Component Gradiometry for isolating common         ║
# ║                 disturbances between sensors                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d
import warnings

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

"Algorithm Parameters"
UFlen=None 
order=1 # 1, 2 or 3
PCA_intervals=None

def clean(B, triaxial = True, sens=0):
    """
    Perform Principal Component gradiometry PCA on the magnetic field data.
    For each order (disturbance type) an optimum analysis interval and scale
    should be supplied through the PCA_intervals and UFlen parameters. 
    Note that for higher orders, the disturbances variance directions 
    (polarizations) should be mutually orthogonal. Moreover, the variance of the
    disturbance during the analysis interval should be larger than the ambient 
    field variation.
    
    If this assumption is violated the quality of the result will be adversely affected.

    Parameters
    ----------
    B : ndarray (n_sensors, axes, n_samples)
        Magnetic field measurements from the sensor array.
    triaxial : bool, optional
        The default is True.
    sens : int, optional
        The sensor number. The default is 0.

    Raises
    ------
    Exception
        If triaxial is False or if order > 3.

    Returns
    -------
    ndarray  (axes, n_samples)
        Corrected measurements corresponding to sensor sens.

    """
    
    if(triaxial == False):
        raise Exception("'triaxial' is set to False. PiCoG only works for triaxial data")
    if order > 3:
        raise Exception('Maximum order is 3')
    
    Bc=np.copy(B)
    if PCA_intervals is None:
        PCA_i={'begin': [0]*3, 'end': [-1]*3}
        warnings.warn('The entire interval will be used to determine'
                      'the coupling matrix. PiCoG.'
                      'PCA_intervals should be used to select optimum intervals.')
    else:
        PCA_i=PCA_intervals
    UF=UFlen if UFlen else [uf]*3
    for odr in range(order):
        ibeg=PCA_i['begin'][odr]
        iend=PCA_i['end'][odr]
        _1, A0, _2 = get_Amtx(np.copy(Bc[:,:,ibeg:iend]), sens=0, uflen=UF[odr])
        _1, A1, _2 = get_Amtx(np.copy(Bc[:,:,ibeg:iend]), sens=1, uflen=UF[odr])
        B0=clean_single_order(Bc, A0, sens=0)
        B1=clean_single_order(Bc, A1, sens=1)
        Bc=np.stack((B0,B1))
    return Bc[sens]

def clean_single_order(B, A=None, sens=0):
    """
    Cleans single order. Uses input matrix A if supplied, otherwise derives it
    using get_Amtx().
    """
    if A is None:
        if PCA_intervals is None:
            PCA_i={'begin': [0]*3, 'end': [-1]*3}
            warnings.warn('The entire interval will be used to determine'
                          'the coupling matrix. PiCoG.'
                          'PCA_intervals should be used to select optimum intervals.')
        else:
            PCA_i=PCA_intervals
        UF=UFlen if UFlen else [uf]*3
        ibeg=PCA_i['begin'][order-1]
        iend=PCA_i['end'][order-1]
        _1, A, _2 = get_Amtx(np.copy(B[:,:,ibeg:iend]), sens=sens, uflen=UF[order-1])
    
    Delta_B = B[1] - B[0]
    B_corrected = B[sens] + A @ Delta_B
    return B_corrected

def get_Amtx(B, sens=0, Detrend=True, uflen=200):
    """
    Derive the coupling matrix A. In addition returns also the scaling factor alpha, 
    the rotation matrices to the Variance Principal Systems (VPS), and the 
    corresponding corrected measurements. 

    Parameters
    ----------
    B : ndarray (n_sensors, axes, n_samples)
        Magnetic field measurements from the sensor array.
    sens : int, optional
        The sensor number. The default is 0.
    Detrend : bool, optional
        Remove the lower frequency band using a uniform filter before analysis. 
        The default is True.
    uflen : int, optional
        Uniform filter length. The default is 200.

    Returns
    -------
    B_corrected : ndarray  (axes, n_samples)
        Corrected measurements corresponding to sensor sens.
    A : ndarray (axes x axes)
        The coupling matrix A.
    dict :
        keys:
            'alpha': the scaling factor alpha 
            'Rsen': the rotation matrix to the sensor VPS
            'Rdif': the rotation matrix to the difference VPS

    """
    B_=np.copy(B)
    if(Detrend):
        trend = uniform_filter1d(B, size=uflen, axis = -1)
        B = B - trend
    
    # Difference between the measurements from the two sensors
    Delta_B = B[1] - B[0]

    # Find VPS coordinate system of Delta_B
    pca = PCA(n_components=3)
    pca.fit(Delta_B.T)
    max_var_dir = pca.components_[0]
    Delta_B_rot, rotation_deltab = rotate_data(Delta_B, max_var_dir)
    r1 = rotation_deltab.as_matrix()
    
    # Find VPS coordinate system of B sensor
    pca = PCA(n_components=3)
    pca.fit(B[sens].T)
    max_var_dir = pca.components_[0]
    B_rot, rotation_b0 = rotate_data(B[sens], max_var_dir)
    r2 = rotation_b0.as_matrix()

    # get the scaling factor
    alpha = np.sqrt(np.var(B_rot[0]) / np.var(Delta_B_rot[0]))
    _plus=B_rot[0]+alpha*Delta_B_rot[0]
    _minus=B_rot[0]-alpha*Delta_B_rot[0]
    if np.var(_plus) < np.var(_minus): alpha=-alpha
    
    # get the transformation matrix
    A=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            A[i,j]=-alpha*np.linalg.inv(r2)[i,0] * r1[0,j]

    # apply the correction
    Delta_B = B_[1] - B_[0]
    B_corrected = B_[sens] + A @ Delta_B
       
    return B_corrected, A, {'alpha':alpha, 'Rsen':r2, 'Rdif':r1}

def rotate_data(data, vector):
    """
    Rotate the data such that the X axis aligns with the provided vector.
    Input:
        data: array of tri-axial data with shape (axes, samples)
        vector: array specifying the vector direction with shape (3,)
    Output:
        rotated_data: rotated version of the input data
    """

    # Normalize the input vector
    vector = vector / np.linalg.norm(vector)

    # Compute the axis of rotation
    axis = np.cross([1, 0, 0], vector)
    axis = axis / np.linalg.norm(axis)

    # Compute the angle of rotation
    angle = np.arccos(vector[0])

    # Define the rotation
    rotation = R.from_rotvec(-angle * axis)

    # Apply the rotation to the data
    rotated_data = rotation.apply(data.T).T

    return rotated_data, rotation