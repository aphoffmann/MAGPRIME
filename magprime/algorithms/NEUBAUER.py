"""
Author: Alex Hoffmann
Last Update: 10/09/2024
Description: Implementation of "Optimization of Multimagnetometer Systems on a Spacecraft" by F. M. Neubauer

General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data

Algorithm Parameters
----------
spacecraft_center : (3,) array of the spacecraft center
mag_positions : (n_sensors, 3) array of the positions of the magnetometers

"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from sklearn.decomposition import PCA

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

"Algorithm Parameters"
spacecraft_center = None # (3,) array of the spacecraft center
mag_positions = None # (n_sensors, 3) array of the positions of the magnetometers
optimize_center = False # boolean for whether to optimize the spacecraft center

def clean(B, triaxial = True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxis: boolean for whether to use triaxial or uniaxial ICA
    """
    if(mag_positions is None or spacecraft_center is None):
        raise("NEUBAUER.mag_positions and NEUBAUER.spacecraft_center must be set before calling clean()")
    
    if(detrend):
        trend = uniform_filter1d(B, size=uf, axis = -1)
        B = B - trend

    result = cleanNeubauer(B)

    if(detrend):
        result += np.mean(trend, axis=0)

    return(result)

def cleanNeubauer2(B):
    n_sensors, axes, n_samples = B.shape

    # Compute rho coefficients and sorted indices
    mat_6b, sorted_indices = compute_mat_6b(spacecraft_center, mag_positions)  # (n_sensors,)

    # Sort B according to sorted_indices
    B_sorted = B[sorted_indices, :, :]  # (n_sensors, axes, n_samples)

    det_mat6b = np.linalg.det(mat_6b)
    inv_mat6b = np.linalg.inv(mat_6b)
    adj_mat6b = det_mat6b * inv_mat6b.T
    C_col1 = adj_mat6b[:, 0] 

    B_amb = np.tensordot(C_col1, B_sorted, axes=([0], [0])) / det_mat6b

    return B_amb

def cleanNeubauer(B):
    """
    Cleans the magnetic field data using the Neubauer method.

    Parameters:
    - B: (n_sensors, axes, n_samples) array of observed magnetic fields.

    Returns:
    - B_amb: (axes, n_samples) array of ambient magnetic fields.
    """
    global spacecraft_center
    if optimize_center:
        result = minimize(
        interference_cost,
        spacecraft_center,
        args=(np.copy(B), mag_positions),
        method='Powell',  # Simplex method suitable for non-smooth functions
        options={'maxiter': 100, 'disp': True})

        # Final spacecraft center estimate
        spacecraft_center = result.x
        print("Optimized: ", result.x)

    n_sensors, axes, n_samples = B.shape

    # Compute rho coefficients and sorted indices
    mat_6b, sorted_indices = compute_mat_6b(spacecraft_center, mag_positions)  # (n_sensors,)

    # Sort B according to sorted_indices
    B_sorted = B[sorted_indices, :, :]  # (n_sensors, axes, n_samples)

    # Compute the ambient magnetic field
    B_amb = np.zeros((axes, n_samples))
    mat_6a = np.ones((n_sensors, n_sensors))
    mat_6a[1:, 1:] = mat_6b[1:, 1:]
    for sample in range(n_samples):
        for axis in range(axes):
            mat_6a[:, 0] = B_sorted[0, axis, sample]
            B_amb[axis, sample] = (1 / np.linalg.det(mat_6b)) * np.linalg.det(mat_6a)

    return B_amb

def interference_cost(spacecraft_center, B_obs, mag_positions):
    """
    Cost function to minimize interference in magnetic field measurements.
    
    Parameters:
    - spacecraft_center: Current estimate of spacecraft center (3,)
    - B_obs: Observed magnetic field data (n_sensors, axes, n_samples)
    - mag_positions: Magnetometer positions (n_sensors, 3)
    
    Returns:
    - cost: Scalar value representing the total interference
    """
    n_sensors, axes, n_samples = B_obs.shape

    # Compute rho_k with the current spacecraft center estimate
    mat_6b, sorted_indices = compute_mat_6b(spacecraft_center, mag_positions)  # (n_sensors,)
    
    # Sort B according to sorted_indices
    B_sorted = B_obs[sorted_indices, :, :]  # (n_sensors, axes, n_samples)

    # Compute the ambient magnetic field
    B_amb = np.zeros((axes, n_samples))
    mat_6a = np.ones((n_sensors, n_sensors))
    mat_6a[1:, 1:] = mat_6b[1:, 1:]
    for sample in range(n_samples):
        for axis in range(axes):
            mat_6a[:, 0] = B_sorted[0, axis, sample]
            B_amb[axis, sample] = (1 / np.linalg.det(mat_6b)) * np.linalg.det(mat_6a)
    
    # Estimate Cost
    interference = B_sorted[-1] - B_sorted[0]
    
    # Works kind of well
    cost = 1/ np.log(np.sum(interference.flatten() - B_amb.flatten()) ** 2)
    
    # Try projection
    #cost = pca_cost(interference.flatten(), B_amb.flatten())
    
    return cost

def pca_cost(interference, B_amb):
    window_length = len(interference) // 20
    traj_interference = [interference[i:i+window_length] for i in range(0,int(window_length * np.floor(len(interference)/window_length)),window_length)] 
    traj_B_amb = [B_amb[i:i+window_length] for i in range(0,int(window_length * np.floor(len(B_amb)/window_length)),window_length)]

    pca = PCA(n_components=1)
    pca.fit(traj_interference)
    x = pca.components_[0]

    pca = PCA(n_components=1)
    pca.fit(traj_B_amb)
    y = pca.components_[0]

    angle = np.abs(np.arccos(np.dot(y, x) / np.linalg.norm(y) / np.linalg.norm(x)) % np.pi)
    return angle


def compute_mat_6b(spacecraft_center = spacecraft_center, mag_positions = mag_positions):
    """
    Computes the rho_i,k coefficients for each component and sensor.
    
    Parameters:
    - mag_positions: (n_sensors, 3) array of magnetometer positions.
    - spacecraft_center: (3,) array of the spacecraft center.
    
    Returns:
    - rho: (3, n_sensors) array where rho[i, k] corresponds to rho_{i,k}
    - sorted_indices: indices that sort the magnetometers for each axis i
    """
    # Compute relative positions
    pos = mag_positions - spacecraft_center  # (n_sensors, 3)

    # Compute distances from spacecraft center
    n_sensors = mag_positions.shape[0]
    mat_6b = np.zeros((n_sensors, n_sensors))

    pos = mag_positions - spacecraft_center  # (n_sensors, 3)
    r_k = np.linalg.norm(pos, axis=1) 

    sorted_indices = np.argsort(-r_k)
    r_sorted = r_k[sorted_indices]

    r1 = r_sorted[0]
    rho_k = r1 / r_sorted

    # Form the rho matrix
    mat_6b[0, :] = 1
    mat_6b[:, 0] = 1
    for i in range(1, n_sensors):
        mat_6b[1:, i] = rho_k[1:] ** (2 + i)

    return mat_6b, sorted_indices