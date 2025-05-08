   #************************************************************************************
   #
   #                           
   #
   #  Program:      spacecraft_spherical_harmonics.py
   #
   #  Programmer:   Dr. Alex Paul Hoffmann
   #                NASA Goddard Space Flight Center
   #                Greenbelt, Maryland  20771
   #
   #  Date:         February 28, 2025
   #
   #  Language:     Python 3
   #
   #  Version:      1.0
   #
   #  Description:  Calculates G and H spherical harmonics coefficients for any arbitrary 
   #                grid of magnetometer measurements. Based on Dumond and Berge (2012).
   #
   #  Notes:        Data must be in the form of (x,y,z) coordinates and (Bx,By,Bz) 
   #                measurements.
   #                
   #                If the magnetometers are stationary, ensure that they are rotated
   #                with respect to the angle spin of the table the spacecraft is on.
   #
   #  Requirements: cvxpy, scipy, pandas, numpy
   #
   #************************************************************************************


import pandas as pd
import numpy as np
import pandas as pd
from scipy.special import assoc_legendre_p  # New function for associated Legendre polynomials
import cvxpy as cp

__all__ = ['characterize_sh']

def p_schmidt_new(l, m, x):
    """
    Compute Schmidt (or geophysically normalized) associated Legendre polynomial P_l^m(x)
    using SciPy's assoc_legendre_p. 
    Set norm=True for normalized polynomials.
    """
    # x = cos(theta)
    if m == 0:
        # Factor = sqrt(2 / (2l+1))
        factor = np.sqrt(2.0 / (2 * l + 1))
    else:
        # Factor = sqrt(4 / (2l+1)) = 2 / sqrt(2l+1)
        factor = 2.0 / np.sqrt(2 * l + 1)

    return np.squeeze(assoc_legendre_p(l, m, x, norm=True)) * factor

def dp_dtheta_analytical(l, m, theta):
    """
    Compute the derivative d/dtheta of the normalized associated Legendre function,
    using a centered finite difference.
    """
    if m == 0:
        # Factor = sqrt(2 / (2l+1))
        factor = np.sqrt(2.0 / (2 * l + 1))
    else:
        # Factor = sqrt(4 / (2l+1)) = 2 / sqrt(2l+1)
        factor = 2.0 / np.sqrt(2 * l + 1)
    
    x = np.cos(theta)
    # Here, diff_n=1 tells assoc_legendre_p to return the first derivative with respect to x.
    dp_dx = assoc_legendre_p(l, m, x, norm=True, diff_n=1)[-1]*factor
    return -np.sin(theta) * dp_dx

def find_sh(x, y, z, Bx, By, Bz, l_max, a=1.0, predict=False):
    """
    Compute Gauss coefficients from B and predict magnetic field components.
    
    Parameters:
    - X,Y,Z: Positions of magnetometer measurements in spacecraft body frame
    - Bx, By, Bz: Magnetic field components at these positions
    - l_max: Maximum integer degree of spherical harmonics 
    - a: Reference radius (must be less than min(r), just use 1.0)
    - predict: If True, return DataFrame with predicted B components
    
    Returns:
    - coefs: List of (l, m, g, h)
    - df_pred: DataFrame with predicted 'x', 'y', 'z', 'Bx', 'By', 'Bz'
    """

    # Convert to spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    if np.any(r < a):
        raise ValueError("All points must have r > a")
    
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Convert observed B components to spherical coordinates (Br, Btheta, Bphi)
    B_r = Bx * sin_theta * np.cos(phi) + By * sin_theta * np.sin(phi) + Bz * cos_theta
    B_theta = Bx * cos_theta * np.cos(phi) + By * cos_theta * np.sin(phi) - Bz * sin_theta
    B_phi = -Bx * np.sin(phi) + By * np.cos(phi)

    # Define coefficient indices: g for all m>=0, and h for m>0.
    coef_list = [(l, m, 'g') for l in range(1, l_max + 1) for m in range(l + 1)] + \
                [(l, m, 'h') for l in range(1, l_max + 1) for m in range(1, l + 1)]
    n_coefs = len(coef_list)

    # Build design matrix for B_r, B_theta, B_phi.
    N = len(x)
    A_r = np.zeros((N, n_coefs))
    A_theta = np.zeros((N, n_coefs))
    A_phi = np.zeros((N, n_coefs))

    for j, (l, m, t) in enumerate(coef_list):
        # Compute normalized associated Legendre polynomial
        p_lm = p_schmidt_new(l, m, cos_theta)
        
        # Use the analytic factor for external field:
        factor = (a / r)**(l + 2)
        if t == 'g':
            trig = np.cos(m * phi)
        else:
            trig = np.sin(m * phi)

        # B_r design matrix
        A_r[:, j] = -(l + 1) * factor * p_lm * trig

        # B_theta design matrix
        dp_dtheta = dp_dtheta_analytical(l, m, theta)
        A_theta[:, j] = factor * dp_dtheta * trig

        # B_phi design matrix
        if m > 0:
            A_phi[:, j] = -m*factor * (p_lm / (sin_theta + 1e-15)) * (np.sin(m * phi) if t=='g' else -np.cos(m * phi))
        else:
            A_phi[:, j] = 0.0

    # Stack the design matrices for B_r, B_theta, B_phi for computation
    A_full = np.concatenate([A_r, A_theta, A_phi])
    b_full = np.concatenate([B_r, B_theta, B_phi])
    
    # Set up the optimization with CVXPY:
    # Define combined coefficient vector
    total_coeffs = cp.Variable(len(coef_list))

    # Regularization (to ensure uniqueness and avoid trivial solutions)
    objective = cp.Minimize(cp.norm(A_full @ total_coeffs - b_full,1) + cp.norm(total_coeffs, 1))

    # Solve the problem
    prob = cp.Problem(objective)
    prob.solve(verbose=False, tol_feas = 1e-20)

    # Extract solved coefficients
    coefs_vector = total_coeffs.value

    # Save coefficients in a dictionary for easy access
    coef_dict = {(l, m): {'g': 0.0, 'h': 0.0} for l in range(1, l_max + 1) for m in range(l + 1)}
    for j, (l, m, t) in enumerate(coef_list):
        if t == 'g':
            coef_dict[(l, m)]['g'] = coefs_vector[j]
        else:  # t == 'h'
            coef_dict[(l, m)]['h'] = coefs_vector[j]

    if predict:
        # Predict field components from these coefficients
        B_r_pred =  A_r @ coefs_vector
        B_theta_pred =  A_theta @ coefs_vector
        B_phi_pred =  A_phi @ coefs_vector

        # Convert spherical to Cartesian
        Bx_pred = B_r_pred * sin_theta * np.cos(phi) + B_theta_pred * cos_theta * np.cos(phi) - B_phi_pred * np.sin(phi)
        By_pred = B_r_pred * sin_theta * np.sin(phi) + B_theta_pred * cos_theta * np.sin(phi) + B_phi_pred * np.cos(phi)
        Bz_pred = B_r_pred * cos_theta - B_theta_pred * sin_theta

        B_est_df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'Bx': Bx_pred, 'By': By_pred, 'Bz': Bz_pred})

        return coef_dict, B_est_df
    else:
        return coef_dict