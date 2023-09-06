
"""
Authors: Alex Hoffmann, 
Date: 04/21/2023
Description: Testing Ream Gradiometry
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

def clean(B0, triaxis = True):
    """
    Perform Principal Component gradiometry PCA on the magnetic field data
    Input:
        B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    Output:
        result: reconstructed ambient field without the spacecraft-generated fields (axes, n_samples)
    """

    if(triaxis == False):
        raise("'triaxis' is set to False. PiCoG only works for triaxial data")

    # Initialize an array to store the corrected magnetic field data
    B_corrected = np.zeros(B0.shape[1:])
    n_sensors = B0.shape[0]

    # Loop over the sensor pairs
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            # Compute the difference between the measurements from the two sensors
            Delta_B0 = B0[i] - B0[j]

            # Perform principal component analysis on Delta_B0
            pca = PCA(n_components=3)
            pca.fit_transform(Delta_B0.T)

            # Get the maximum variance direction of Delta_B0
            max_var_dir = pca.components_[0]

            # Project Delta_B0 onto the maximum variance direction
            Delta_B0_x, rotation = rotate_data(Delta_B0, max_var_dir)

            # Project B0[i] onto the maximum variance direction
            B0_i_x, _ =  rotate_data(B0[i], max_var_dir)

            # Estimate the scaling factor alpha using the variance ratio
            alpha = np.sqrt(np.var(B0_i_x[0]) / np.var(Delta_B0_x[0]))

            # Correct the maximum variance component of B0[i] using alpha and Delta_B0_x
            B_corrected[0] = B0[i][0] - alpha * Delta_B0_x[0]

            B_corrected = rotation.inv().apply(B_corrected.T).T

    # Return the corrected magnetic field data
    return B_corrected


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
