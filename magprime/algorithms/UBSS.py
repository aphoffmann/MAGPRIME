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


"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = False     # Detrend the data

"Algorithm Parameters"
sigma = 100         # Magnitude Filter Threshold
lambda_ = 1.2       # Magnitude Filter Threshold Factor
sspTol = 15         # SSP Filter Threshold
bpo = 10            # Number of Bands Per Octave in the NSGT Transform
fs = 1              # Sampling Frequency
weight = 1          # Weight for Compressive Sensing
boom = None         # Index of boom magnetometer in (n_sensors, axes, n_samples) array
cs_iters = 5        # Number of Iterations for Compressive Sensing
subspace = False    # Use Subspace Method for Compressive Sensing

"Internal Parameters"
magnetometers = 3
result = None
clusterCentroids = collections.OrderedDict({0:
                       np.ones(magnetometers) })
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
        for axis in range(3):
            setMagnetometers(B.shape[0])
            clusterNSGT(B[:,axis,:])
            result[axis] = demixNSGT(B[:,axis,:])[0]
    else:
        setMagnetometers(B.shape[0])
        clusterNSGT(B)
        result = demixNSGT(B)[0]

    if(detrend):
        result += np.mean(trend, axis=0)
    
    return(result)

def subspacePursuit2(A, b, n_clusters, data):
    """
    Implement Subspace Pursuit algorithm using CVXPY for complex-valued data.
    
    Parameters:
        A : ndarray
            Measurement matrix (m x n), complex-valued.
        b : ndarray
            Observed data vector (m x 1), complex-valued.
        n_clusters : int
            Sparsity level (number of non-zero entries in the signal).
        data : ndarray
            Data to set b.
    Returns:
        x : ndarray
            Reconstructed signal (n x 1), complex-valued.
        support : ndarray
            Support indices of the reconstructed signal.
    """
    # Ensure b is set to data
    y = data.copy()
    Phi = A.value
    s = n_clusters
    mu = 2  # You can adjust mu as needed or pass it as a parameter

    # Initialization
    S = set()  # S^0 = empty set
    x = np.zeros(Phi.shape[1], dtype=complex)  # x^0 = 0
    max_iters = 100  # Maximum number of iterations
    tol = 1e-6       # Convergence tolerance

    for n in range(max_iters):
        # Step 1: Compute Phi^*(y - Phi x^{n-1})
        residual = y - Phi @ x
        proxy = Phi.conj().T @ residual

        # Step 2: Identify Delta S (s largest entries in proxy)
        Delta_S = set(np.argsort(np.abs(proxy))[-s:])

        # Step 3: Merge support sets
        S_union = S.union(Delta_S)

        # Step 4: Solve least squares on the merged support
        if len(S_union) == 0:
            x_tilde = np.zeros(Phi.shape[1], dtype=complex)
        else:
            Phi_S = Phi[:, list(S_union)]
            # Using least squares: minimize ||y - Phi_S z||_2
            z, _, _, _ = np.linalg.lstsq(Phi_S, y, rcond=None)
            x_tilde = np.zeros(Phi.shape[1], dtype=complex)
            x_tilde[list(S_union)] = z

        # Step 5: Identify U^n (s largest entries in x_tilde)
        U = set(np.argsort(np.abs(x_tilde))[-s:])

        # Step 6: Form u^n by keeping entries in U and zeroing out others
        u = np.zeros(Phi.shape[1], dtype=complex)
        u[list(U)] = x_tilde[list(U)]

        # Step 7: Update support set S^n
        residual_u = y - Phi @ u
        Phi_residual = Phi.conj().T @ residual_u
        update_term = u + mu * Phi_residual
        S_new = set(np.argsort(np.abs(update_term))[-s:])

        # Step 8: Solve least squares on the new support set
        if len(S_new) == 0:
            x_new = np.zeros(Phi.shape[1], dtype=complex)
        else:
            Phi_S_new = Phi[:, list(S_new)]
            z_new, _, _, _ = np.linalg.lstsq(Phi_S_new, y, rcond=None)
            x_new = np.zeros(Phi.shape[1], dtype=complex)
            x_new[list(S_new)] = z_new

        # Check for convergence
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            S = S_new
            print(f'Converged at iteration {n+1}')
            break

        # Update variables for next iteration
        x = x_new
        S = S_new
    else:
        print('Maximum iterations reached without convergence.')


    # Optional: Enforce additional constraints if applicable
    # Example: Enforce boom constraint (if 'boom' is defined)
    # if 'boom' in globals() and boom is not None:
    #     if np.abs(x[0]) >= np.abs(x[boom]):
    #         x[0] = x[boom]
    
    return x


def subspacePursuit(A, b, n_clusters, data):
    """
    Implement Subspace Pursuit algorithm using CVXPY for complex-valued data.
    
    Parameters:
        A : ndarray
            Measurement matrix (m x n), complex-valued.
        b : ndarray
            Observed data vector (m x 1), complex-valued.
        n_clusters : int
            Dimension of the signal x (number of variables).
        data : ndarray
            Data to set b.
    Returns:
        x : ndarray
            Reconstructed signal (n x 1), complex-valued.
    """
    # Perform SSP check
    b_real = np.real(data)
    b_imag = np.imag(data)
    cos_sim = np.dot(b_real, b_imag) / (np.linalg.norm(b_real) * np.linalg.norm(b_imag))
    threshold = np.cos(np.deg2rad(sspTol))
    SSP = cos_sim >= threshold
    
    # Adjust K based on SSP
    if SSP:
        return processData(A, b, n_clusters, data)
    else:
        K = A.shape[0]

    # Ensure b is set to data
    b = data.copy()
    A = A.value
    
    # Initialize residual
    r = b.copy()
    
    # Initialize support set S
    S = np.array([], dtype=int)

    
    # Iterative process
    for iter in range(cs_iters):
        # Compute proxy
        p = A.conj().T @ r
        
        # Identify indices of largest K entries in |p|
        indices = np.argsort(np.abs(p))[-K:]
        
        # Merge with current support set
        S_candidate = np.union1d(S, indices)
        
        # Solve least squares problem over support set using CVXPY
        A_S = A[:, S_candidate]
        
        lambda_reg = 1e-3
        A_S_conj_transpose = A_S.conj().T
        lhs = A_S_conj_transpose @ A_S + lambda_reg * np.eye(len(S_candidate))
        rhs = A_S_conj_transpose @ b

        # Solve for x_S
        try:
            x_S = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error at iteration {iter}: {e}")
            break  # Exit loop or handle the error as needed

        # Prune support to keep largest K components
        idx_largest = np.argsort(np.abs(x_S))[-K:]
        S = S_candidate[idx_largest]
        x_S = x_S[idx_largest]

        # Update residual
        A_S = A[:, S]
        r = b - A_S @ x_S

        # Check convergence
        if np.linalg.norm(r) < 1e-6:
            break

    
    # Form the full x vector
    x = np.zeros(n_clusters, dtype=complex)
    x[S] = x_S
    
    # Enforce boom constraint if applicable
    if boom is not None and np.abs(x[0]) >= np.abs(b[boom]):
        x[0] = b[boom]
    
    return x


def processData(A, b, n_clusters, data):
    "Define cvxpy parameters and variables for optimization problem" 
    x = cp.Variable(shape = n_clusters, complex=True)
    
    weights = np.ones(n_clusters)/n_clusters; 
    w = cp.Parameter(shape = n_clusters, value = weights, nonneg=True)
    
    "Define constraints as Dantzig Selector and optional boom constraint"
    constraints = [cp.norm(A.T@(A@x - b), 'inf') <= 0.008]
    
    "Define objective function as weighted L1 norm"
    objective = cp.Minimize(cp.sum(w.T@cp.abs(x)))
        
    "Instantiate Problem"     
    problem = cp.Problem(objective, constraints)
    b.value = data

    "Check if Single Source Point"          
    b_real = np.real(data); b_imag = np.imag(data)
    cos_sim = np.dot(b_real, b_imag) / (np.linalg.norm(b_real) * np.linalg.norm(b_imag))
    threshold = np.cos(np.deg2rad(sspTol))
    SSP = cos_sim >= threshold
    
    x_ratio = 0
    
    "Iteratively solve the system" 
    for i in range(cs_iters):
        try:
            problem.solve(warm_start=True)
            if(problem.status == 'optimal'): break
        except:
            "Check if x is None"
            if(x.value is None):
                x.value = np.zeros(n_clusters)
                x.value[0] = b.value[np.abs(b.value).argmin()]

            #string = f"ECOS Solver Failed\nASSP: {ASSP}\nX: {x.value}\nW: {w.value}\nB: {b.value}\nA: {A.value}\nRatio: {x_ratio}\n status: {problem.status}"
            #raise Exception(string)
        

        if(SSP): 
            "Make W[0] Smaller"
            w = cp.inv_pos(cp.abs(x) + 0.01)
        else:
            delta = calculate_delta_s(A.value, x.value)
            if(delta < np.sqrt(2) - 1):
                w = cp.inv_pos(cp.abs(x) + 0.01)
                continue
            else:
                "Calculate signal to noise ratio"
                x_hat = np.abs(x.value) 
                x_ratio = np.sum(x_hat[1:])/( x_hat[0]+ 0.01)
                
                "Update and clip ambient field weight"
                w.value[0] = w.value[0] + .1*(x_ratio - w.value[0])
                w.value[0]  = np.clip(w.value[0], .01, 100)

    "Check if boom constraint is violated"
    if(boom and np.abs(x.value[0]) >= np.abs(b.value[boom])):
        x.value[0] = b.value[boom]

    return x.value
     
def weightedReconstruction(sig):
    "Get the number of clusters and initialize an empty result array"
    n_clusters = len(clusterCentroids)
    result = np.zeros((sig.shape[1], n_clusters), dtype = 'complex_')
    
    "Convert the cluster centroids and signal to numpy arrays"
    centroids = np.array([clusterCentroids[i] for i in clusterCentroids.keys()])
    s = sig.T
    s = np.array(s)
    
    "Define CVXPY parameters"
    A = cp.Parameter(shape=centroids.T.shape, value=centroids.T, complex=True)
    #print(np.round(A.value,2))
    b = cp.Parameter(shape = magnetometers, complex=True)   
    
    "Pack constants together"
    if(subspace):
        func = partial(subspacePursuit, A, b, n_clusters)
    else:
        func = partial(processData, A, b, n_clusters)
    
    "Use multiprocessing pool to map processData function over s array"
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use imap_unordered and tqdm
        result = list(tqdm.tqdm(pool.imap(func, s), total=np.max(s.shape)))
    
    r=np.array(result).T
    return(r)
   
def setMagnetometers(n=3):
    "Set the number of magnetometers"
    global magnetometers; global clusterCentroids
    magnetometers = n
    clusterCentroids = collections.OrderedDict({0:
                           np.ones(n) })
    
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
    magFilter = m > lambda_*sigma
    B_m = np.array([B[i][magFilter] for i in range(magnetometers)])
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
    B_s = np.array([B[i][SSP_Bools] for i in range(magnetometers)])
    return(B_s) 
    
def clusterNSGT(sig):
    "Create instance of NSGT and set NSGT parameters"
    length = sig.shape[-1]
    bins = bpo
    fmax = fs/2
    lowf = 8 * bpo * fs / length
    nsgt = CQ_NSGT(lowf, fmax, bins, fs, length, multichannel=True)
        
    "Take Non-stationary Gabor Transform"
    B = nsgt.forward(sig)
    B = np.array(B, dtype=object)
    B = np.vstack([np.hstack(B[i]) for i in range(magnetometers)])
    
    "Filter Low Energy Points"
    B_m = filterMagnitude(B)

    "Filter Single Source Points"
    B_ssp = filterSSP(B_m)
    
    "Take Absolute Magnitude"
    B_abs = np.abs(B_ssp)
    
    "Find Cos and Sin of Argument"
    B_ang = np.abs(np.angle(B_ssp) - np.angle(B_ssp[0]))
    B_cos, B_sin = np.cos(B_ang), np.sin(B_ang)
    
    "Project to Unit Hypersphere and Join with Argument"
    norms = np.sqrt((B_abs**2).sum(axis=0,keepdims=True))
    B_projected = np.where(norms!=0,B_abs/norms,0.)
    H_tk =  np.vstack([B_projected,B_cos, B_sin])
        
    "Cluster Data"
    (centroids, clusters) = clusterData(H_tk)
    
    "Find Gain and Phase"
    gain = centroids[:,:magnetometers]
    B_cos = centroids[:,magnetometers:2*magnetometers]
    B_sin = centroids[:,2*magnetometers:]
    phase = np.arctan2(B_sin, B_cos)
    
    "Normalize Gain"
    norms = np.sqrt((gain**2).sum(axis=1,keepdims=True))
    gain = np.where(norms!=0,gain/norms,0.)
    
    "Form Mixing Matrix"
    mixingMatrix = gain *  np.exp(1j*phase)
     
    "Update Global Mixing Matrix"
    updateCentroids(mixingMatrix.T)
    return

"""Define a function to demix a signal using non-stationary Gabor transform (NSGT)"""
def demixNSGT(sig):
    "Create instance of NSGT and set NSGT parameters"
    length = sig.shape[-1]
    bins = bpo
    fmax = fs/2
    lowf = 8 * bpo * fs / length
    nsgt = CQ_NSGT(lowf, fmax, bins, fs, length, multichannel=True)
    
    "Apply the forward transform to the signal and convert to numpy array"
    B = nsgt.forward(sig)
    B = np.array(B, dtype=object)
    
    "Get the shapes of each subband in each channel"
    shapes = np.array([i.shape[-1] for i in B[0]])
    
    "Stack and concatenate the subbands from each channel into a matrix"
    B_nsgt = np.vstack([np.hstack(B[i]) for i in range(magnetometers)])
    
    "Separate Signals"
    B_reconstructed = weightedReconstruction(B_nsgt)

        
    "Split the matrix into subbands for each channel"
    S_nsgt = []
    for arr in B_reconstructed:
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
    
    return(result)
 
def updateCentroids(newCentroids, learnRate = 0.1):
    "Check if Clusters are in the global mixing matrix"
    if newCentroids.T.size > 0: ## Check if no new centroids
        for centroid in newCentroids.T:
            
            newC = True
            for cluster in clusterCentroids:
                a = np.real(clusterCentroids[cluster]) / np.linalg.norm(clusterCentroids[cluster]);
                b = np.real(centroid)/np.linalg.norm(centroid)
                angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
                if(angle < np.deg2rad(sspTol)):
                    if(cluster != 0):
                        clusterCentroids[cluster] = clusterCentroids[cluster] + learnRate * (centroid - clusterCentroids[cluster])
                    newC = False
            
            "Add New Cluster"        
            if(newC):
                clusterCentroids[len(clusterCentroids)] = centroid
    return(np.array([clusterCentroids[i] for i in clusterCentroids.keys()]))
    
"UTILITY FUNCTIONS"   
def frequencyPlot(F, title="Frequency Plot", hypersphere = False, plot_density = False, pm = False):
    fig = plt.figure()
    fig.suptitle(title)
 
    x,y,z = F[0],F[1],F[2]
    
    if(plot_density):
        xyz = np.vstack([F[0],F[1],F[2]])
        density = gaussian_kde(xyz)(xyz) 
        idx = density.argsort()
        x, y, z, density = x[idx], y[idx], z[idx], density[idx]
    
    ax = fig.add_subplot(projection='3d')
    if(hypersphere):
        n_theta = 50 # number of values for theta
        n_phi = 200  # number of values for phi
        r = 1       #radius of sphere
        theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        ax.plot_wireframe(x, y, z, rstride=10, cstride=10, color='grey')
    
    if(plot_density):
        ax.scatter(x, y, z, c=density)
    else:
        ax.scatter(x, y, z)
    ax.set_xlabel('B*(t,k)', fontsize = '12')
    ax.set_ylabel('B*(t,k)', fontsize = '12')
    ax.set_zlabel('B*(t,k)', fontsize = '12') 
    ax.tick_params(labelsize='8' )
        
    plt.show()
   
def rip_check(A, k=1, p=1):
    # A: numpy matrix
    # k: sparsity level
    # p: norm parameter
    m, n = A.shape # get matrix dimensions
    delta = 0 # initialize RIP constant
    for i in range(n): # loop over all columns
        x = np.zeros(n) # create a zero vector
        x[i] = 1 # set one entry to one (unit vector)
        indices = np.random.choice(n, k-1, replace=False) # choose k-1 random indices
        x[indices] = np.random.randn(k-1) # set those entries to random values (sparse vector)
        ratio = np.linalg.norm(A @ x, p) / np.linalg.norm(x, p) # compute ratio of norms
        delta = max(delta, abs(ratio - 1)) # update RIP constant if ratio is larger than previous value
    return delta # return RIP constant

def calculate_delta_s(A, x):
    # A: sensing matrix
    # x: signal estimate (vector)
    # Calculate the norm of A @ x and x
    Ax_norm = np.linalg.norm(A @ x, 2)
    x_norm = np.linalg.norm(x, 2)
    
    # Calculate the ratio of the norms squared
    ratio = (Ax_norm / x_norm) ** 2
    
    # Estimate the RIP constant delta_s for the current support of x
    # It's the maximum deviation of the ratio from 1
    delta_s = max(abs(ratio - 1), abs(1 - ratio))
    
    return delta_s