"""
Author: XYZ, Alex Hoffmann
Last Update: 9/19/2023
Description: Todo

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import uniform_filter1d

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

def clean(B, triaxial = True):
    """
    Perform Principal Component gradiometry PCA on the magnetic field data
    Input:
        B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    Output:
        result: reconstructed ambient field without the spacecraft-generated fields (axes, n_samples)
    """
    if(detrend):
        trend = uniform_filter1d(B, size=uf, axis = -1)
        B -= trend
    
    if(triaxial == False):
        raise Exception("'triaxial' is set to False. PiCoG only works for triaxial data")

    result = clean_first_order(B)

    if(detrend):
        result += np.mean(trend, axis=0)

            
    # Return the corrected magnetic field data
    return result

def clean_first_order(B):
    B_corrected = np.zeros(B.shape[1:])
    
    # Compute the difference between the measurements from the two sensors
    Delta_B = B[1] - B[0]

    # Perform principal component analysis on Delta_B
    pca = PCA(n_components=3)
    pca.fit_transform(Delta_B.T)

    # Get the maximum variance direction of Delta_B
    max_var_dir = pca.components_[0]

    # Project Delta_B and B[i] onto the maximum variance direction
    Delta_B_rot, rotation = rotate_data(Delta_B, max_var_dir)
    B_rot, _ =  rotate_data(B[0], max_var_dir)

    # Estimate the scaling factor alpha using the variance ratio
    alpha =  np.sqrt(np.var(Delta_B[0]) / np.var(B_rot[0]))

    # Correct the maximum variance component of B0[i] using alpha and Delta_B0_x
    B_corrected[0] = B_rot[0] - alpha * Delta_B[0]
    B_corrected[1] = B_rot[1]
    B_corrected[2] = B_rot[2]

    B_corrected = rotation.inv().apply(B_corrected.T).T

    # Return the corrected magnetic field data
    return B_corrected

def clean_higher_order(B, order = 2):
    B_new = np.copy(B)
    for i in range(order):
        B_0 = clean_first_order(B_new)
        B_1 = clean_first_order(B_new[[1,0]])
        B_new = np.stack((B_0,B_1))
    return(B_0)



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

    # Compute the angle of rotation
    angle = np.arccos(np.dot([1, 0, 0], vector))

    # Define the rotation
    rotation = R.from_rotvec(angle * axis)

    # Apply the rotation to the data
    rotated_data = rotation.apply(data.T).T

    return rotated_data, rotation
