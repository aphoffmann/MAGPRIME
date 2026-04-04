# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  RUDEST.py                                                   ║
# ║  Package      :  magprime                                                    ║
# ║  Author       : Marissa Verduin  <marissa.f.verduin@nasa.gov                 ║
# ║                 Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>                 ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-08-08                                                  ║
# ║  Last Updated :  2025-08-08                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  :                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import numpy as np
import magpylib as magpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from scipy.linalg import hankel
from scipy.signal import hilbert
from scipy.fft import fft
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


def detect(X, p, q, stride_x, stride_y, nu_value, n_components=3, template_moment=None, template_height=None):
    '''
    Run the main functions to create map of anomalies from the inputted image/data and parameters
    '''

    #call build_template function if template_moment and template_height are inputted
    if template_moment is None or template_height is None:
        X_template = None
    else:
        X_template = build_template(p, q, template_moment, template_height)

    #Create trajectory matrix
    [T,pos] = build_T(X, p, q, stride_x, stride_y, X_template)

    #Perform PCA and OC-SVM
    [principal_components, y_pred] = PCA_SVM(T, nu_value, n_components)

    #Create map of anomalous points
    A = anomaly_map(X, p, q, y_pred, pos)

    #Filter map by the distribution of coverage over the data
    A_filtered = filter(X,p,q,stride_x,stride_y, A)

    return([principal_components, y_pred, X, A, A_filtered])

def plot(principal_components, y_pred, X, A, A_filtered, x, y, n_components):
    '''
    Plot the principal components and the comparison between the original data and anomaly detection output
    '''
    #Principal Components Plot#
    plt.figure(figsize=(12,8))

    #create 3D plot for PC if input n_components=3 (default)
    if n_components ==3: 
        plt.figure(figsize=(12,8))
        plt.scatter(principal_components[:,0],principal_components[:,1], c=y_pred, cmap='bwr')
        plt.title('OC-SVM Clustered PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        anomaly = Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=10)
        nominal = Line2D([0], [0], marker='o', color='w', label='Nominal', markerfacecolor='blue', markersize=10)
        plt.legend(handles=[anomaly, nominal], loc='upper left')

        plt.figure(figsize=(14,8))
        ax = plt.subplot(111, projection='3d')
        plt.scatter(principal_components[:,0],principal_components[:,1],principal_components[:,2], c=y_pred, cmap='bwr')
        plt.title('OC-SVM Clustered PCA')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Prinicipal Component 3')
        anomaly = Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=10)
        nominal = Line2D([0], [0], marker='o', color='w', label='Nominal', markerfacecolor='blue', markersize=10)
        plt.legend(handles=[anomaly, nominal], loc='upper left')
        ax.set_box_aspect(None, zoom=0.85)
        plt.show()
    #Create 2D plot for PC
    else:
        plt.scatter(principal_components[:,0],principal_components[:,1], c=y_pred, cmap='bwr')
        plt.title('OC-SVM Clustered PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        anomaly = Line2D([0], [0], marker='o', color='w', label='Anomaly', markerfacecolor='red', markersize=10)
        nominal = Line2D([0], [0], marker='o', color='w', label='Nominal', markerfacecolor='blue', markersize=10)
        plt.legend(handles=[anomaly, nominal], loc='upper left')
        plt.show()

    #X and A_filtered Plot#
    axes = [np.min(x),np.max(x),np.min(y),np.max(y)]

    #Image Data
    plt.figure(figsize=(12,8))
    plt.subplot(121)
    plt.imshow(X, extent=axes, cmap='nipy_spectral', origin='lower')
    plt.title('Lunar Surface Magnetic Field')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.grid(visible=True)
    plt.colorbar(shrink=0.6, label='Magnetic Field Intensity (T)')

    #Anomaly Map
    plt.subplot(122)
    plt.imshow(A_filtered>0.5, cmap='nipy_spectral', extent=axes, origin='lower')
    plt.title('Anomaly Map')
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.grid(visible=True)
    plt.colorbar(shrink=0.6, label='Anomaly Detection')
    plt.show()

    return()

def build_template(p, q, template_moment, template_height):
    '''
    Create template of magnetic field data for filtering (if template moment and template height are inputted)

    template_dipole : magnetic dipole
    ts : template spacing
    template_grid : base grid for template data
    B_template : calculates B for the template grid
    X_template : template of B magnitudes
     '''
    
    #create magnetic dipole based on the moment
    template_dipole = magpy.misc.Dipole(moment=template_moment)

    #create template dimensions and grid
    ts = np.linspace(-.5, .5, p)
    template_grid = np.array([[(x, y, template_height) for x in ts] for y in ts])

    #create B values and take magnitude for template
    B_template = template_dipole.getB(template_grid)   
    B_template = np.moveaxis(B_template, -1, 0)
    X_template = np.linalg.norm(B_template, axis=0)

    return(X_template)

def build_T(X, p, q, stride_x, stride_y, X_template=None):
    """
    Call vectorize function to create the trajectory matrix and compare to template

    X : Grid of data
    p : Number of rows in patch
    q : Number of columns in patch
    stride_x : Patch movement in the x-direction
    stride_y : Patch movement in the y-direction
    template : Template p-by-q image 
    n : Number of patches
    S : Vector of each patch
    T : Trajectory matrix
    pos : index for each patch location
    template_S : sigma matrix for the template (if input template)
    T_final : dot product of T and template (if input template)
    """

    #Initialize variables
    T = [] 
    pos = []

    #Shift p-by-q window with x & y strides
    for i in range(0, X.shape[0]-p+1, stride_y):
        for j in range(0, X.shape[1]-q+1, stride_x):
            S = vectorize(X,p,q,j,i)
            T.append(S) #trajectory matrix

            pos.append((i,j)) #index position of top left point in each patch
    
    #Compare trajectory matrix to template if inputted
    T = np.array(T)
    
    if X_template is None: #skip steps if no template
        return([T,pos])
    else: 
        template_S = np.squeeze(vectorize(X_template,p,q))
        T_final = T * template_S
        return([np.array(T_final),pos])

def vectorize(X, p, q, column_start=1, row_start=1):
    """
    Create one row for each p-by-q patch

    column_start : Increment of each patch section for the columns (X-direction)
    row_start : Increment of each patch section for the rows (Y-direction)
    H : Demeaned Hankel-block resultant for each flattened patch
    C : Covariance matrix
    C_unique : Unique values of covariance matrix
    circulant : circulant matrix
    y : Singular Value Vector 
    """

    #Singular Value Decomposition (SVD)
    H = hankelize(X, p, q, column_start, row_start)

    C = H*np.transpose(H) #Find the covariance matrix    
    C /= p*q #normalize
    
    C_unique = np.append(C[:,0], C[-1,1:-1]) #only take first column and last row of unique values
    C_unique = np.abs(C_unique) #magnitude (remove imaginary components)

    circulant = np.append(C_unique,np.flip(C_unique))

    y = fft(circulant) #singular values from Fast Fourier Transform
    y = np.abs(y[:len(C)]) #truncated

    return(y)

def hankelize(X, p, q, column_start, row_start):
    """    
    Create demeaned Hankel matrix

    h : Width of hankel block
    patch_flat : Flattened array of the entire patch
    patch : Analytic signal computation for the patch
    H : Hankel-block resultant for each flattened patch
    H_flat : Flattened array of the hankel matrix
    mean : Mean of each row in Hankel block
    H_demean : Demeaned Hankel block
    """
    #calculate width of hankel matrix based on window size
    h = int(((p*q)+1)/2)

    #create Hankel matrix from flattened patch
    patch_flat = X[row_start:p+row_start,column_start:column_start+q].flatten() 
    patch = hilbert(patch_flat) #filter out negative frequencies
    H = hankel(patch[:h],patch[-h:])

    #Demean the hankel matrix
    H_flat = H.flatten()
    mean = np.mean(H_flat)
    H_demean = H - mean  

    return(H_demean)

def PCA_SVM(T, nu_value, n_components=3):
    '''
    Run data through PCA and OC-SVM to determine principal components, cluster data, and flag anomalies

    features : trajectory matrix
    pca : Principal Components Analysis
    prinicipal components : components describing variance in data
    ocsvm : One-Class Support Vector Machine
    y_pred : predicted anomalies
    '''
    features = T

    # Principal Component Analysis (PCA) - Dimension Reduction
    pca = PCA(n_components)
    principal_components = pca.fit_transform(features)
 
    # One-Class Support Vector Machine (SVM) - Clustering
    ocsvm = svm.OneClassSVM(nu=nu_value, kernel='rbf', gamma='scale')
    ocsvm.fit(principal_components)

    # Feature Detection #
    # Predict anomalies: 1 for normal, -1 for anomalies
    y_pred = ocsvm.predict(principal_components)

    # Floor predictions at 0 for easy summation analysis
    y_pred = np.where(y_pred < 0, 1, 0)

    return([principal_components, y_pred])

def anomaly_map(X, p, q, y_pred, pos):
    """
    Create mapping matrix

    A : matrix of anomalies
    """
    #initialize variable
    A = np.zeros(np.shape(X))

    #Add anomaly label to each point from feature detection
    for (i, j), label in zip(pos, y_pred):
        A[i:i+p, j:j+q] += label 

    return(A)

def filter(X, p, q, stride_x, stride_y, A):
    """
    Create distribution of coverage from x and y strides to filter output

    coverage : distribution of analysis per point
    A_filtered : anomaly matrix with coverage filter applied
    """
    #initialize variable
    coverage=np.zeros(X.shape)

    #create matrix of coverage for each point in build_T
    for i in range(0, X.shape[0]-p+1, stride_y):
        for j in range(0, X.shape[1]-q+1, stride_x): 
            for k in range(q):
                    coverage[i:p+i, k+j] += 1 

    A_filtered = A / coverage #remove any bias from less coverage on the edges

    return(A_filtered)
