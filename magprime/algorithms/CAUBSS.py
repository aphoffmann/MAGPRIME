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
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
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
boom = None         # Index of boom magnetometer in (n_sensors, axes, n_samples) array
cs_iters = 5        # Number of Iterations for Compressive Sensing

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
        result = np.zeros((3, B.shape[-1]))
        setMagnetometers(B.shape[0])
        clusterNSGT(B)
        # result[axis] = demixNSGT(B[:,axis,:])[0] ToDO
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
    magFilter = m > np.percentile(m, 95)
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

def processData(A_big, b_big, n_clusters, data):
    """
    A_big : cp.Parameter, shape = (n_sensors*n_axes, n_clusters*n_axes)
    b_big : cp.Parameter, shape = (n_sensors*n_axes,)
    data  : a 2D measurement in original shape => we flatten it before passing here,
            or pass it already flattened in weightedReconstruction.
    
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
    x_2d = cp.reshape(x_big, (n_clusters, n_axes))

    # Dantzig-type constraint: norm(A.T @ (A@x - b), inf) <= 0.01
    # Here, A_big@x_big => shape (n_sensors*n_axes,)
    # b_big             => shape (n_sensors*n_axes,)
    constraints = [
        cp.norm(A_big.T @ (A_big @ x_big - b_big), 'inf') <= 0.01
    ]

    # Weighted L1 norm on x_2d
    # w^T@|x_2d| => shape mismatch unless we sum across axes
    # Your original code is cp.sum(w.T @ cp.abs(x)),
    # which works if cp.abs(x) is (n_clusters, 3) and w is (n_clusters,).
    # That yields a shape (3,) expression. Usually you'd sum again, but let's match your code style:
    #objective = cp.Minimize(cp.sum(w.T @ cp.abs(x_2d)))
    #objective = cp.Minimize(cp.sum(cp.multiply(w[:, None], cp.abs(x_2d))))
    objective = cp.Minimize(
        cp.sum(cp.multiply(w, cp.norm(x_2d, p=2, axis=1)))
    )
    problem = cp.Problem(objective, constraints)

    # Assign the flattened measurement
    b_big.value = data  # must be shape (n_sensors*n_axes,)

    # Solve
    problem.solve(warm_start=True)

    return x_big.value

def weightedReconstruction(sig):
    """
    Performs the reconstruction for each 'frame' (or slice) of `sig`
    using a parallel pool. The result is shaped accordingly.
    
    sig : shape (n_sensors, num_samples, n_axes) or something similar
          so that sig.T might get you an iterable over the time dimension.
    """
    # 1) Gather cluster centroids as an array of shape (n_clusters, n_sensors, n_axes)
    n_clusters = len(clusterCentroids)
    centroids = np.array([clusterCentroids[i] for i in clusterCentroids.keys()])
    # Now centroids has shape (n_clusters, n_sensors, n_axes)

    # 2) Build the block-diagonal version of A
    A_big_np = build_block_diag_matrix(centroids)
    # shape => (n_sensors * n_axes,  n_clusters * n_axes)
    
    # Create CVXPY parameters for A_big and b_big
    A_big = cp.Parameter(shape=A_big_np.shape, complex=True, value=A_big_np)
    b_big = cp.Parameter(shape=(magnetometers * 3,), complex=True)

    # Check shapes
    print("A_big shape =", A_big.value.shape)

    # 3) Prepare data array
    #  sig is presumably shape (magnetometers, something, 3),
    #  so sig.T => shape (something, magnetometers, 3).
    #  We'll iterate over each slice in 's'.
    s = np.transpose(sig, (2, 0, 1))  # shape => (# frames, magnetometers, 3) if sig was (magnetometers, #frames, 3)
    s = np.array(s)

    # 4) We'll pass partial(...) with the processData function
    func = partial(processData, A_big, b_big, n_clusters)

    # 5) Use multiprocessing
 
    results = []
    for frame in tqdm.tqdm(s, total=len(s)):
        # Flatten the frame to shape (magnetometers*3,)
        flat_frame = frame.reshape(-1)
        # Call processData directly with the flattened frame
        result = func(flat_frame)
        results.append(result)
    """
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm.tqdm(pool.imap(func, 
                                           # Flatten each slice to shape (magnetometers*3,)
                                           (frame.reshape(-1) for frame in s),
                                           # or you can do direct pass if you already shaped them
                                          ), 
                                 total=len(s)))  
    """

    # 6) Convert results to desired shape:
    # Each 'result' is x_big, shape (n_clusters*3,)
    # We might want shape (#frames, 3, n_clusters), or something else
    results = np.array(results)  # shape (#frames, n_clusters*3)
    
    # We'll reshape each row to (n_clusters,3), then transpose as needed
    # If you want final shape (sig.shape[2], 3, n_clusters) like your original code:
    #   sig.shape[2] might be #frames. Let's assume #frames = results.shape[0].
    #   Then each result row => (n_clusters, 3).
    #   Then we store it in shape (#frames, 3, n_clusters).
    # This is an example—adjust if you want a different layout:
    reshaped = []
    for row in results:
        # row => (n_clusters*3,)
        x_2d = row.reshape(n_clusters, 3)  # shape (n_clusters, 3)
        x_2d = x_2d.T                     # shape (3, n_clusters)
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
    r_flat = B_reconstructed.reshape(sensors*axes, B_reconstructed.shape[-1])

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
    result = result.reshape(sensors, axes, result.shape[-1])
    
    return(result)
 