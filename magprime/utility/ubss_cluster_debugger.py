"""
Author: Alex Hoffmann
Last Update: 3/13/2024
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
import hdbscan
import collections
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import itertools
from nsgt import CQ_NSGT

hdbscan = hdbscan.HDBSCAN(min_samples = 4)

param_ranges = {
    'sigma': list(range(1, 5000, 50)),
    'lambda_': list(range(1, 2, 1)),
    'sspTol': list(range(5, 25, 5)),
    'bpo': [1, 2, 4, 8, 16]
}

def clusterData(B):
    "Cluster Samples x M data points on unit hypersphere"
    clusterData = B.T
    clusters = hdbscan.fit_predict(clusterData)
    labels = hdbscan.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    C = [clusterData[labels == i] for i in range(n_clusters_)]
    centroids = [np.mean(C[i], axis=0) for i in range(n_clusters_)]
    centroids = np.round(np.matrix(centroids), 3)
    return centroids, clusters, clusterData

def filterMagnitude(B, lambda_, sigma):
    """ Filters out low energy points"""
    B = np.array(B)
    m = np.linalg.norm(np.abs(B), axis=0)
    magFilter = m > lambda_ * sigma
    B_m = np.array([B[i][magFilter] for i in range(B.shape[0])])
    return B_m

def filterSSP(B, sspTol):
    """Filter out Multi Source Points"""
    a = np.array(np.real(B))
    b = np.array(np.imag(B))
    a_dot_b = (a * b).sum(axis=0)
    norm_a = np.atleast_1d(np.linalg.norm(a, 2, 0))
    norm_a[norm_a == 0] = 1
    norm_b = np.atleast_1d(np.linalg.norm(b, 2, 0))
    norm_b[norm_b == 0] = 1
    cos_sim = np.abs(a_dot_b / (norm_a * norm_b))
    SSP_Bools = np.array(np.matrix(cos_sim >= np.cos(np.deg2rad(sspTol)))).flatten()
    B_s = np.array([B[i][SSP_Bools] for i in range(B.shape[0])])
    return B_s

def clusterNSGT(sig, sigma, lambda_, sspTol, bpo, fs, plot = False):
    "Create instance of NSGT and set NSGT parameters"
    length = sig.shape[-1]
    magnetometers = sig.shape[0]
    fmax = fs / 2
    lowf = 2 * bpo * fs / length
    nsgt = CQ_NSGT(lowf, fmax, bpo, fs, length, multichannel=True)

    "Take Non-stationary Gabor Transform"
    B = nsgt.forward(sig)
    B = np.array(B, dtype=object)
    B = np.vstack((np.hstack(B[i]) for i in range(magnetometers)))

    "Filter Low Energy Points"
    B_m = filterMagnitude(B, lambda_, sigma)

    "Filter Single Source Points"
    B_ssp = filterSSP(B_m, sspTol)

    "Take Absolute Magnitude"
    B_abs = np.abs(B_ssp)

    "Find Cos and Sin of Argument"
    B_ang = np.abs(np.angle(B_ssp) - np.angle(B_ssp[0]))
    B_cos, B_sin = np.cos(B_ang), np.sin(B_ang)

    "Project to Unit Hypersphere and Join with Argument"
    norms = np.sqrt((B_abs ** 2).sum(axis=0, keepdims=True))
    B_projected = np.where(norms != 0, B_abs / norms, 0.)
    H_tk = np.vstack((B_projected, B_cos, B_sin))

    if plot:
        frequencyPlot(B_projected, title="B_projected", hypersphere = True, plot_density = True, pm = False)

    "Cluster Data"
    centroids, clusters, clustered_data = clusterData(H_tk)

    "Find Gain and Phase"
    gain = centroids[:, :magnetometers]
    B_cos = centroids[:, magnetometers:2 * magnetometers]
    B_sin = centroids[:, 2 * magnetometers:]
    phase = np.arctan2(B_sin, B_cos)

    "Normalize Gain"
    norms = np.sqrt((gain ** 2).sum(axis=1, keepdims=True))
    gain = np.where(norms != 0, gain / norms, 0.)

    "Form Mixing Matrix"
    mixingMatrix = gain * np.exp(1j * phase)

    "Update Global Mixing Matrix"
    return centroids, clusters, clustered_data, mixingMatrix

def find_optimal_parameters(data, fs, param_ranges):
    best_params = None
    best_silhouette = -1
    best_calinski = -1

    param_combinations = itertools.product(*param_ranges.values())

    for params in param_combinations:
        sigma, lambda_, sspTol, bpo = params

        # Run your clustering algorithm with the given parameters
        try:
            centroids, clusters, clustered_data, mixingMatrix = clusterNSGT(data, sigma, lambda_, sspTol, bpo, fs)
        except:
            print(f"Error with parameters: {params}")
            continue
        
        # Compute cluster validity indices
        silhouette = silhouette_score(clustered_data, clusters)
        calinski = calinski_harabasz_score(clustered_data, clusters)

        # Check if the current parameter combination gives better scores
        if silhouette > best_silhouette or (silhouette == best_silhouette and calinski > best_calinski):
            best_params = (sigma, lambda_, sspTol, bpo)
            best_silhouette = silhouette
            best_calinski = calinski

    centroids, clusters, clustered_data, mixingMatrix = clusterNSGT(data, best_params[0],best_params[1],best_params[2],best_params[3], fs)

    return best_params, best_silhouette, best_calinski



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

