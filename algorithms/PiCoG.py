
"""
Authors: Alex Hoffmann, 
Date: 04/21/2023
Description: Testing Ream Gradiometry
"""

import numpy as np
from sklearn.decomposition import PCA


# Define a function to clean the magnetic field data
def clean(B0):
    # B0 is a 3D array of shape (n_sensors, n_components, n_samples)
    # containing the measured magnetic field data from n_sensors sensors
    # along n_components axes over n_samples time points

    # Initialize an array to store the corrected magnetic field data
    B_corrected = np.zeros_like(B0)
    n_sensors = B0.shape[0]
    # Loop over the sensor pairs
    for i in range(n_sensors):
        for j in range(i+1, n_sensors):
            # Compute the difference between the measurements from the two sensors
            Delta_B0 = B0[i] - B0[j]

            # Perform principal component analysis on Delta_B0
            pca = PCA(n_components=3)
            pca.fit(Delta_B0.T)

            # Get the maximum variance direction of Delta_B0
            max_var_dir = pca.components_[0]

            # Project Delta_B0 onto the maximum variance direction
            Delta_B0_x = np.dot(Delta_B0.T, max_var_dir)

            # Perform principal component analysis on B0[i]
            pca = PCA(n_components=3)
            pca.fit(B0[i].T)

            # Get the maximum variance direction of B0[i]
            max_var_dir_i = pca.components_[0]

            # Project B0[i] onto the maximum variance direction
            B0_i_x = np.dot(B0[i].T, max_var_dir_i)

            # Estimate the scaling factor alpha using the variance ratio
            alpha = np.var(B0_i_x) / np.var(Delta_B0_x)

            # Correct the maximum variance component of B0[i] using alpha and Delta_B0_x
            B_corrected[i][0] = B0[i][0] - alpha * Delta_B0_x

            # Copy the other two components of B0[i] without correction
            B_corrected[i][1:] = B0[i][1:]

    # Return the corrected magnetic field data
    return B_corrected



# Define a function to clean the magnetic field data with higher order corrections
def clean_higher_order(B0):
  # B0 is a 3D array of shape (n_sensors, n_components, n_samples)
  # containing the measured magnetic field data from n_sensors sensors
  # along n_components axes over n_samples time points

  # Initialize an array to store the corrected magnetic field data
  B_corrected = np.zeros_like(B0)

  # Loop over the sensor pairs
  for i in range(n_sensors):
    for j in range(i+1, n_sensors):

      # Compute the difference between the measurements from the two sensors
      Delta_B0 = B0[i] - B0[j]

      # Perform principal component analysis on Delta_B0
      pca = PCA(n_components=3)
      pca.fit(Delta_B0.T)

      # Get the maximum variance direction of Delta_B0
      max_var_dir = pca.components_[0]

      # Project Delta_B0 onto the maximum variance direction
      Delta_B0_x = np.dot(Delta_B0.T, max_var_dir)

      # Perform principal component analysis on B0[i]
      pca = PCA(n_components=3)
      pca.fit(B0[i].T)

      # Get the maximum variance direction of B0[i]
      max_var_dir_i = pca.components_[0]

      # Project B0[i] onto the maximum variance direction
      B0_i_x = np.dot(B0[i].T, max_var_dir_i)

      # Estimate the scaling factor alpha using the variance ratio
      alpha = np.var(B0_i_x) / np.var(Delta_B0_x)

      # Correct the maximum variance component of B0[i] using alpha and Delta_B0_x
      B_corrected[i][0] = B0[i][0] - alpha * Delta_B0_x

      # Copy the other two components of B0[i] without correction
      B_corrected[i][1:] = B0[i][1:]

      # Initialize an array to store the higher order correction matrices
      A = np.zeros((n_sensors, n_sensors, n_components, n_components))

      # Compute the higher order correction matrices for each sensor pair
      for n in range(1, n_sensors):

        # Compute the rotation matrix from the sensor system to the VPS of B_corrected[n-1]
        R_n_i = pca.fit_transform(B_corrected[n-1][i].T).T

        # Compute the rotation matrix from the sensor system to the VPS of Delta_B_corrected[n-1]
        R_n_ij = pca.fit_transform((B_corrected[n-1][i] - B_corrected[n-1][j]).T).T

        # Compute the elements of A[n-1] using Eq. (17)
        for k in range(n_components):
          for l in range(n_components):
            A[n-1][i][j][k][l] = -alpha * R_n_i[k][0] * R_n_ij[0][l]

        # Apply the higher order correction to B_corrected[n]
        B_corrected[n][i] = B_corrected[n-1][i] + np.dot(A[n-1][i][j], Delta_B_corrected[n-1])

  # Return the corrected magnetic field data
  return B_corrected



# Define the function clean(B) that takes a matrix B of magnetic field measurements as input
def clean2(B):
  # Initialize an empty list to store the cleaned measurements
  B_clean = []
  # Loop over each sensor pair i,j
  for i in range(len(B)):
    for j in range(i+1, len(B)):
      # Compute the difference between the measurements from the two sensors
      Delta_B = B[i] - B[j]
      # Perform principal component analysis on the difference matrix
      pca = PCA()
      pca.fit(Delta_B)
      # Get the maximum variance direction and component
      max_var_dir = pca.components_[0]
      max_var_comp = pca.transform(Delta_B)[:,0]
      # Compute the correction matrix A using Eq. (17)
      alpha = np.var(B[i] @ max_var_dir) / np.var(max_var_comp)
      R_i = pca.components_.T
      R_ij = R_i @ pca.components_
      A = -alpha * R_i[:,0] * R_ij[0,:]
      # Apply the correction to the measurements from sensor i using Eq. (16)
      B_i_corrected = B[i] + A @ Delta_B.T
      # Append the corrected measurements to the list
      B_clean.append(B_i_corrected)
  # Return the list of cleaned measurements
  return B_clean
