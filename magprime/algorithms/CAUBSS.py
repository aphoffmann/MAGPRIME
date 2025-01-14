"""
Author: Alex Hoffmann
Last Update: 9/19/2023
Description: 
             
General Parameters
----------
uf : window size for uniform filter used to detrend the data
detrend : boolean for whether to detrend the data

Algorithm Parameters
----------
sigma : magnitude filter threshold
lambda_ : magnitude filter threshold factor
sspTol : SSP filter threshold
bpo : Number of Bands Per Octave in the NSGT Transform
fs : sampling frequency
weight : weight for compressive sensing
boom : index of boom magnetometer in (n_sensors, axes, n_samples) array
"""

import numpy as np
from sklearn.cluster import HDBSCAN
import cvxpy as cp
import collections
import multiprocessing as mp
from nsgt import CQ_NSGT
import tqdm
from functools import partial
from scipy.ndimage import uniform_filter1d
from scipy.linalg import block_diag


"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

"Algorithm Parameters"
sspTol = 15         # SSP Filter Threshold
bpo = 1            # Number of Bands Per Octave in the NSGT Transform
fs = 1              # Sampling Frequency
mag_percentile = 95
mag_positions = None

"Internal Parameters"
magnetometers = 3
result = None
clusterCentroids = collections.OrderedDict({0:
                       np.ones((magnetometers,3)) })
hdbscan = HDBSCAN(min_samples = 4)

def clean(B, triaxial = True):
    """
    Perform magnetic noise removal through Underdetermined Blind Source Separation
    Input:
        B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    Output:
        result: reconstructed ambient field without the spacecraft-generated fields (axes, n_samples)
    """
    if(detrend):
        trend = uniform_filter1d(B, size=uf, axis = -1)
        B = B - trend

    if(triaxial):
        setMagnetometers(B.shape[0])
        clusterNSGT(B)
        result = demixNSGT(B)[0]
    else:
        raise Exception("Only triaxial data is supported")

    if(detrend):
        result += np.mean(trend, axis=0)
    
    return(result)

def setMagnetometers(n=3):
    "Set the number of magnetometers"
    global magnetometers; global clusterCentroids
    magnetometers = n
    clusterCentroids = collections.OrderedDict({0:
                       np.ones((magnetometers,3)) })

def clusterData(B):
    "Cluster Samples x M data points on unit hypersphere"
    clusterData = B.T
    clusters = hdbscan.fit_predict(clusterData)
    labels = hdbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    C = [clusterData[labels == i] for i in range(n_clusters_)]
    centroids = [np.mean(C[i], axis=0) for i in range(n_clusters_)]
    centroids = np.round(np.matrix(centroids),3)
    return(centroids, clusters)       
      
def filterMagnitude(B):
    """ Filters out low energy points"""
    B = np.array(B)
    m = np.linalg.norm(np.abs(B), axis=0)
    magFilter = m > np.percentile(m, mag_percentile)
    B_m = np.array([B[i][magFilter] for i in range(B.shape[0])])
    return(B_m)

def filterSSP(B):
    """Filter out Multi Source Points"""
    a = np.array(np.real(B))
    b = np.array(np.imag(B))
    a_dot_b = (a*b).sum(axis=0)
    norm_a = np.atleast_1d(np.linalg.norm(a, 2, 0))
    norm_a[norm_a==0] = 1
    norm_b = np.atleast_1d(np.linalg.norm(b, 2, 0))
    norm_b[norm_b==0] = 1
    cos_sim = np.abs(a_dot_b/(norm_a*norm_b))
    SSP_Bools = np.array(np.matrix(cos_sim >= np.cos(np.deg2rad(sspTol)))).flatten()
    B_s = B[:,SSP_Bools]
    return(B_s) 
    
def updateCentroids(newCentroids, learnRate=0.1, sspTol=15):
    """
    newCentroids shape: (n_sensors, n_clusters, axes)
      - n_sensors: number of sensors
      - n_clusters: number of new centroids to consider
      - axes: number of axes per sensor (e.g., 3 for triaxial)

    clusterCentroids is a global OrderedDict where each entry is:
      clusterCentroids[cluster_id] -> np.array shape (n_sensors, axes)
    """
    n_sensors, n_clusters, axes = newCentroids.shape
    
    # If no new centroids, just return the current list
    if n_clusters == 0:
        return np.array([clusterCentroids[i] for i in clusterCentroids.keys()])
    
    # Loop over each new centroid in the second dimension
    for i in range(n_clusters):
        # Extract the (n_sensors, axes) centroid
        centroid_2d = newCentroids[:, i, :]  # shape: (n_sensors, axes)
        
        # Flatten for cosine-similarity comparison
        centroid_flat = centroid_2d.flatten()
        
        # Normalize to handle magnitude differences
        # (Take the real part in case it’s complex, though typically these are real)
        centroid_norm = np.linalg.norm(centroid_flat)
        if centroid_norm > 0:
            centroid_flat = np.real(centroid_flat) / centroid_norm
        
        newC = True  # Flag indicating whether this centroid is brand new
        
        # Compare with all existing centroids
        for cluster_id, existing_centroid_2d in clusterCentroids.items():
            # Flatten and normalize existing centroid
            existing_flat = existing_centroid_2d.flatten()
            existing_norm = np.linalg.norm(existing_flat)
            if existing_norm > 0:
                existing_flat = np.real(existing_flat) / existing_norm
            
            # Cosine similarity = dot(a, b); we then compute angle = arccos(...)
            dot_val = np.dot(existing_flat, centroid_flat)
            # Clip to handle floating-point errors slightly outside [-1,1]
            dot_val = np.clip(dot_val, -1.0, 1.0)
            angle = np.arccos(dot_val)  # in radians
            
            # Check if angle is within tolerance
            if angle < np.deg2rad(sspTol):
                # Update the existing centroid via a simple learning rate
                # Usually skip updating cluster=0, but replicate your logic:
                if cluster_id != 0:
                    clusterCentroids[cluster_id] += learnRate * (centroid_2d - clusterCentroids[cluster_id])
                newC = False
                break  # Already found a match; no need to continue
        
        # If after checking all existing centroids this is still new, add it
        if newC:
            clusterCentroids[len(clusterCentroids)] = centroid_2d

    # Return all global centroids in an array for convenience
    return np.array([clusterCentroids[i] for i in clusterCentroids.keys()])

def clusterNSGT(sig):
    "Create instance of NSGT and set NSGT parameters"
    sensors, axes, length = sig.shape
    sig_flat = sig.reshape(sensors*axes, length)
    bins = bpo
    fmax = fs/2
    lowf = 8 * bpo * fs / length
    nsgt = CQ_NSGT(lowf, fmax, bins, fs, length, multichannel=True)
        
    "Take Non-stationary Gabor Transform"
    B = nsgt.forward(sig_flat)
    B = np.array(B, dtype=object)
    B = np.vstack([np.hstack(B[i]) for i in range(sensors*axes)])
    
    "Filter Low Energy Points and Single Source Points"
    B_m = filterMagnitude(B)
    B_ssp = filterSSP(B_m)
    B_ssp = B_ssp.reshape(sensors, axes, B_ssp.shape[-1]) # (sensor, axes, samples)
    
    "Take Absolute Magnitude"
    B_abs = np.abs(B_ssp)
    
    "Find Cos and Sin of Argument"
    B_ang = np.abs(np.angle(B_ssp) - np.angle(B_ssp[0]))
    B_cos, B_sin = np.cos(B_ang), np.sin(B_ang)
    
    "Project to Unit Hypersphere and Join with Argument"
    norms = np.sqrt((B_abs**2).sum(axis=0,keepdims=True))
    B_projected = np.where(norms!=0,B_abs/norms,0.)
    H_tk =  np.vstack([B_projected,B_cos, B_sin]).transpose((1,0,2))
    H_tk = H_tk.reshape(3*sensors*axes, H_tk.shape[-1])
        
    "Cluster Data"
    (centroids, clusters) = clusterData(H_tk)

    "Extract components"
    centroids = centroids.reshape(centroids.shape[0], 3, centroids.shape[1]//3) # Shape: (n_clusters, axes, sensors * axes)
    gain = centroids.T[:sensors].T
    B_cos = centroids.T[sensors:2*sensors].T
    B_sin = centroids.T[2*sensors:].T
    phase = np.arctan2(B_sin, B_cos)

    "Form Mixing Matrix"
    mixingMatrix = gain * np.exp(1j*phase)
    mixingMatrix = mixingMatrix.transpose(2,0,1) # Shape: (n_sensors, n_clusters, axes)
    updateCentroids(mixingMatrix)
    return

def build_block_diag_matrix(centroids):
    """
    Given `centroids` of shape (n_clusters, n_sensors, n_axes),
    build a block-diagonal matrix A_big of shape:
        (n_sensors * n_axes) x (n_clusters * n_axes)
    where each diagonal block is A_axis = (n_sensors x n_clusters)
    for the corresponding axis.
    """
    n_clusters, n_sensors, n_axes = centroids.shape

    blocks = []
    for ax in range(n_axes):
        # centroids[:, :, ax] has shape (n_clusters, n_sensors)
        # we usually want (n_sensors, n_clusters) to multiply by x of shape (n_clusters,)
        A_axis = centroids[:, :, ax].T  # shape => (n_sensors, n_clusters)
        blocks.append(A_axis)
    
    # Block-diagonal stacking
    # Final shape = (n_sensors*n_axes,  n_clusters*n_axes)
    A_big_np = block_diag(*blocks)
    return A_big_np

def build_block_diag_matrix_amb(centroids):
    n_clusters, n_sensors, n_axes = centroids.shape

    blocks = []
    for ax in range(n_axes):
        # centroids[:, :, ax] has shape (n_clusters, n_sensors)
        # we usually want (n_sensors, n_clusters) to multiply by x of shape (n_clusters,)
        A_axis = np.zeros((n_sensors,n_clusters))  # shape => (n_sensors, n_clusters)
        A_axis[:,0] = 1
        blocks.append(A_axis)
    A_big_np = block_diag(*blocks)
    return A_big_np

def buildLaplacianOperator(mag_positions):
    """
    Build a (3M x 3M) matrix L that approximates the Laplacian of the magnetic field
    at each sensor using finite differences along nearest neighbors.

    Parameters
    ----------
    mag_positions : np.ndarray of shape (M, 3)
        The (x, y, z) positions of each of the M magnetometers.

    Returns
    -------
    L : np.ndarray of shape (3*M, 3*M)
        A naive discrete Laplacian operator.
    """
    M = len(mag_positions)
    L = np.zeros((3*M, 3*M), dtype=float)

    # Helper indexers for axis-major B:
    def idx_x(m): return m
    def idx_y(m): return M + m
    def idx_z(m): return 2*M + m

    # For each sensor, build second derivative approximations for each axis
    for axis in range(3):  # 0->x, 1->y, 2->z
        # Sort sensors by coordinate along this axis
        sorted_indices = np.argsort(mag_positions[:, axis])
        
        for idx in sorted_indices:
            # Index conversion for accessing corresponding rows/cols in L
            if axis == 0:
                row = idx_x(idx)
            elif axis == 1:
                row = idx_y(idx)
            else:
                row = idx_z(idx)
            
            # Find neighbors along the axis
            neighbors = []
            # Try to get previous neighbor in sorted order
            prev_idx = np.where(sorted_indices == idx)[0][0] - 1
            if prev_idx >= 0:
                neighbors.append(sorted_indices[prev_idx])
            # Try to get next neighbor in sorted order
            next_idx = np.where(sorted_indices == idx)[0][0] + 1
            if next_idx < len(sorted_indices):
                neighbors.append(sorted_indices[next_idx])
            
            # If no neighbors found, skip Laplacian approximation
            if not neighbors:
                continue
            
            # Use distances for finite difference approximation
            coeff_center = 0.0
            for n in neighbors:
                # Determine column index corresponding to neighbor and axis
                if axis == 0:
                    col = idx_x(n)
                elif axis == 1:
                    col = idx_y(n)
                else:
                    col = idx_z(n)
                
                # Distance between sensor i and neighbor n along current axis
                d = mag_positions[n, axis] - mag_positions[idx, axis]
                if np.isclose(d, 0):
                    continue
                
                # Off-diagonal term for neighbor contribution
                L[row, col] = 1.0 / (d**2)
                coeff_center -= 1.0 / (d**2)
            
            # Center coefficient: sum contributions from neighbors
            L[row, row] = -coeff_center

    return L


def findCurlOperator(mag_positions):
    """
    Build a (3M x 3M) matrix D that, when multiplied by a magnetic field vector
    B of axis-major shape [x1..xM, y1..yM, z1..zM], returns an approximate curl
    at each sensor.

    Parameters
    ----------
    mag_positions : np.ndarray of shape (M, 3)
        The (x, y, z) positions of each of the M magnetometers.

    Returns
    -------
    D : np.ndarray of shape (3*M, 3*M)
        A naive discrete-curl operator. Then C = D @ B has shape (3*M,),
        where C[3*i : 3*i+3] ~ (curl_x, curl_y, curl_z) at sensor i.
    """

    M = len(mag_positions)
    # We want a 3M x 3M operator: 3 curl-components per sensor, 3 field-components per sensor
    D = np.zeros((3*M, 3*M), dtype=float)

    # Helper indexers for axis-major B:
    #   B = [x0, x1, ..., x(M-1), y0, y1, ..., y(M-1), z0, z1, ..., z(M-1)]
    def idx_x(m): return m
    def idx_y(m): return M + m
    def idx_z(m): return 2*M + m

    def find_closest_in_axis(i, axis):
        """
        Return the index j != i of the sensor with the closest coordinate
        to sensor i in the specified axis (0->x,1->y,2->z).
        If there's a tie or degenerate, this just picks the first min.
        """
        coords = mag_positions[:, axis]
        val = coords[i]
        diff = np.abs(coords - val)
        diff[i] = -1  # exclude itself
        j = np.argmax(diff)
        return j

    # For each sensor i, we fill 3 rows of D that approximate:
    #
    #   C_x(i) = (∂Bz/∂y)(i) - (∂By/∂z)(i)
    #   C_y(i) = (∂Bx/∂z)(i) - (∂Bz/∂x)(i)
    #   C_z(i) = (∂By/∂x)(i) - (∂Bx/∂y)(i)
    #
    # Each partial derivative is approximated via a first difference with
    # the nearest neighbor along that axis.

    for i in range(M):
        # -----------------------------
        # Row 3*i => approximate C_x(i) = ∂Bz/∂y - ∂By/∂z
        # -----------------------------
        # ∂Bz/∂y at sensor i
        j_y = find_closest_in_axis(i, axis=1)  # neighbor along y
        dy = mag_positions[j_y,1] - mag_positions[i,1]
        if not np.isclose(dy, 0.0):
            coeff_z_by = 1.0 / dy
            # "Bz(j_y) - Bz(i)" * coeff_z_by
            D[3*i, idx_z(j_y)] +=  coeff_z_by
            D[3*i, idx_z(i)]   -= coeff_z_by
        else:
            # degenerate, skip or treat as 0
            pass

        # ∂By/∂z at sensor i
        j_z = find_closest_in_axis(i, axis=2)  # neighbor along z
        dz = mag_positions[j_z,2] - mag_positions[i,2]
        if not np.isclose(dz, 0.0):
            coeff_y_bz = 1.0 / dz
            # Subtract (∂By/∂z)
            D[3*i, idx_y(j_z)] -= coeff_y_bz
            D[3*i, idx_y(i)]   += coeff_y_bz

        # -----------------------------
        # Row 3*i+1 => approximate C_y(i) = ∂Bx/∂z - ∂Bz/∂x
        # -----------------------------
        # ∂Bx/∂z
        j_z2 = find_closest_in_axis(i, axis=2)
        dz2 = mag_positions[j_z2,2] - mag_positions[i,2]
        if not np.isclose(dz2, 0.0):
            coeff_x_bz = 1.0 / dz2
            D[3*i+1, idx_x(j_z2)] += coeff_x_bz
            D[3*i+1, idx_x(i)]    -= coeff_x_bz

        # ∂Bz/∂x
        j_x = find_closest_in_axis(i, axis=0)
        dx = mag_positions[j_x,0] - mag_positions[i,0]
        if not np.isclose(dx, 0.0):
            coeff_z_bx = 1.0 / dx
            # subtract
            D[3*i+1, idx_z(j_x)] -= coeff_z_bx
            D[3*i+1, idx_z(i)]   += coeff_z_bx

        # -----------------------------
        # Row 3*i+2 => approximate C_z(i) = ∂By/∂x - ∂Bx/∂y
        # -----------------------------
        # ∂By/∂x
        if not np.isclose(dx, 0.0):
            coeff_y_bx = 1.0 / dx
            D[3*i+2, idx_y(j_x)] += coeff_y_bx
            D[3*i+2, idx_y(i)]   -= coeff_y_bx

        # ∂Bx/∂y
        j_y2 = find_closest_in_axis(i, axis=1)
        dy2 = mag_positions[j_y2,1] - mag_positions[i,1]
        if not np.isclose(dy2, 0.0):
            coeff_x_by = 1.0 / dy2
            # subtract
            D[3*i+2, idx_x(j_y2)] -= coeff_x_by
            D[3*i+2, idx_x(i)]    += coeff_x_by

    return D

def processData(A_big, A_big_amb, b_big, n_clusters, D, data):
    """
    A_big : cp.Parameter, shape = (n_sensors*n_axes, n_clusters*n_axes)
    b_big : cp.Parameter, shape = (n_sensors*n_axes,)
    data  : a 2D measurement in original shape => we pass it already flattened in weightedReconstruction.
    
    We'll reconstruct x in (n_clusters, 3) by first defining x_big in (n_clusters*3,).
    """

    # Number of axes we assume is 3
    n_axes = 3
    # A_big.shape[0] = n_sensors*n_axes

    # Define x_big as a 1D variable that we'll reshape
    x_big = cp.Variable(shape=(n_clusters * n_axes,), complex=True)

    # For weighting:
    weights = np.ones(n_clusters)/n_clusters
    w = cp.Parameter(shape=(n_clusters,), value=weights, nonneg=True)

    # We'll reshape x_big back to (n_clusters, 3) for the L1 penalty
    x_2d = cp.reshape(x_big, (n_axes, n_clusters))

    # Dantzig-type constraint: norm(A.T @ (A@x - b), inf) <= 0.01
    constraints = [
        cp.norm(A_big.T @ (A_big @ x_big - b_big), 'inf') <= 0.01,
        cp.norm( D @ A_big_amb @ x_big, 2) <= 0.01,
    ]

    # Weighted L1 norm on x_2d
    objective = cp.Minimize(cp.sum(cp.multiply(w, cp.norm(x_2d, p=1, axis=0))))
    problem = cp.Problem(objective, constraints)

    # Assign the flattened measurement
    b_big.value = data 

    # Solve
    try:
        problem.solve()
    except:
        problem.solve(solver=cp.SCS, verbose = True,)

    # Todo check if larger than smallest

    return x_big.value

def weightedReconstruction(sig):
    """
    Performs the reconstruction for each 'frame' (or slice) of `sig`
    using a parallel pool. The result is shaped accordingly.
    
    sig : shape (n_sensors, n_axes, n_samples) 
    """
    # 1) Gather cluster centroids as an array of shape (n_clusters, n_sensors, n_axes)
    n_clusters = len(clusterCentroids)
    centroids = np.array([clusterCentroids[i] for i in clusterCentroids.keys()])
    # Now centroids has shape (n_clusters, n_sensors, n_axes)

    # 2) Build the block-diagonal version of A
    A_big_np = build_block_diag_matrix(centroids)
    A_big_np_amb = build_block_diag_matrix_amb(centroids)
    # shape => (n_sensors * n_axes,  n_clusters * n_axes)
    
    # Create CVXPY parameters for A_big and b_big
    A_big = cp.Parameter(shape=A_big_np.shape, complex=True, value=A_big_np)
    A_big_amb = cp.Parameter(shape=A_big_np_amb.shape, complex=True, value=A_big_np_amb)
    b_big = cp.Parameter(shape=(magnetometers * 3,), complex=True)

    # 3) Prepare data array
    s = np.transpose(sig, (2, 0, 1))  # shape => (# frames, magnetometers, axes) 
    s = np.array(s)

    # 4) We'll pass partial(...) with the processData function
    if mag_positions is not None:
        D = findCurlOperator(mag_positions)
        D_big = cp.Parameter(shape=D.shape, value=D)
        func = partial(processData,A_big, A_big_amb, b_big, n_clusters, D_big)
    else:
        func = partial(processData, A_big, A_big_amb, b_big, n_clusters, None)
    # 5) Use multiprocessing
    """
    results = []
    for frame in tqdm.tqdm(s, total=len(s)):
        # Flatten the frame to shape (magnetometers*3,)
        flat_frame = frame.T.reshape(-1)
        # Call processData directly with the flattened frame [x1, x2, y1, y2, z1, z2]
        result = func(flat_frame)
        results.append(result)
    """
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(func, 
                                           # Flatten each slice to shape (magnetometers*3,)
                                           (frame.T.reshape(-1) for frame in s),
                                           # or you can do direct pass if you already shaped them
                                          ), 
                                 total=len(s)))  
    

    # 6) Convert results to desired shape:
    results = np.array(results)  # shape (#frames, n_clusters*3)
    
    # We'll reshape each row to (3,n_clusters), then transpose as needed
    reshaped = []
    for row in results:
        x_2d = row.reshape(3, n_clusters)  # shape (n_clusters, 3)
        reshaped.append(x_2d)
    r = np.array(reshaped)  # shape (#frames, 3, n_clusters)

    # If you want to transpose to return shape (n_clusters, 3, #frames), do .transpose
    # but let's just return r as is
    return r.T
   
"""Define a function to demix a signal using non-stationary Gabor transform (NSGT)"""
def demixNSGT(sig):
    "Create instance of NSGT and set NSGT parameters"
    sensors, axes, length = sig.shape
    sig_flat = sig.reshape(sensors*axes, length)
    length = sig.shape[-1]
    bins = bpo
    fmax = fs/2
    lowf = 8 * bpo * fs / length
    nsgt = CQ_NSGT(lowf, fmax, bins, fs, length, multichannel=True)
    
    "Apply the forward transform to the signal and convert to numpy array"
    B = nsgt.forward(sig_flat)
    B = np.array(B, dtype=object)
    
    "Get the shapes of each subband in each channel"
    shapes = np.array([i.shape[-1] for i in B[0]])
    
    "Stack and concatenate the subbands from each channel into a matrix"
    B_nsgt = np.vstack([np.hstack(B[i]) for i in range(sensors*axes)])
    B_nsgt = B_nsgt.reshape(sensors, axes, B_nsgt.shape[-1])
    
    "Separate Signals"
    B_reconstructed = weightedReconstruction(B_nsgt)
    r_flat = B_reconstructed.reshape(len(clusterCentroids)*axes, B_reconstructed.shape[-1])

    "Split the matrix into subbands for each channel"
    S_nsgt = []
    for arr in r_flat:
        index = 0
        sig = []
        for shape in shapes:
            sig.append(arr[index:index + shape])
            index += shape
            
        S_nsgt.append(sig)
    
    "Apply the backward transform to get the demixed signal"
    sig_r = nsgt.backward(S_nsgt)
    
    "Save Result"
    global result
    result = np.array(sig_r)
    result = result.reshape(len(clusterCentroids), axes, result.shape[-1])
    
    return(result)
 