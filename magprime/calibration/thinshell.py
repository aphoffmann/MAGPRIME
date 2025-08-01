import numpy as np
from scipy.optimize import least_squares

__all__ = ["thinshell"]

def thinshell(Bx, By, Bz, ref_B):
    """
    Solve for:
      A      : 3×3 upper‑triangular calibration matrix
      O      : 3‑vector of axis offsets
      gains  : scale factors [s1, s2, s3]
      angles : misalignment [phi, rho, lam] in degrees
    
    on noise‑free data so that ||A*(B - O)|| = ref_B exactly.
    """
    # 1) INITIAL LINEAR SOLVE FOR S = AᵀA AND c = S·O → then Cholesky to get A, O
    #    build D·p = ref_B²  where p = [s0..s5, c0..c2, d]
    D = np.column_stack([
        Bx*Bx,             # s0
        By*By,             # s1
        Bz*Bz,             # s2
        2*Bx*By,           # s3
        2*Bx*Bz,           # s4
        2*By*Bz,           # s5
        -2*Bx,             # c0
        -2*By,             # c1
        -2*Bz,             # c2
    ])
    
    y = ref_B**2
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    s0, s1, s2, s3, s4, s5, c0, c1, c2 = p

    # reconstruct S and c
    S = np.array([[s0, s3, s4],
                  [s3, s1, s5],
                  [s4, s5, s2]])
    c = np.array([c0, c1, c2])
    # offsets:
    O_init = np.linalg.solve(S, c)
    # upper‑triangular A from Cholesky of S:
    L = np.linalg.cholesky(S)
    A_init = L.T

    # extract initial gains & angles (same formulas as before)
    g0, g1, g2 = A_init[0,0], A_init[1,1], A_init[2,2]
    a12, a13, a23 = A_init[0,1], A_init[0,2], A_init[1,2]

    # 2) NONLINEAR REFINEMENT OVER [s1,s2,s3, a12,a13,a23, O0,O1,O2]
    def resid(params, Bx, By, Bz, ref_B):
        s1,s2,s3,a12,a13,a23,O0,O1,O2 = params
        A = np.array([[s1, a12, a13],
                        [0. , s2 , a23],
                        [0. , 0. , s3 ]])
        X = np.vstack((Bx - O0, By - O1, Bz - O2))
        B_cal = A @ X
        B_norm = np.linalg.norm(B_cal, axis=0)
        return B_norm - ref_B

    # pack initial guess
    x0 = np.array([
        g0, g1, g2,          # s1,s2,s3
        a12, a13, a23,       # off‑diagonals
        *O_init              # O0,O1,O2
    ])
    sol = least_squares(resid, x0, args=(Bx,By,Bz,ref_B), loss='huber',
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    

    s1_, s2_, s3_, a12_, a13_, a23_, O0, O1, O2 = sol.x
    A = np.array([[s1_, a12_, a13_],
                  [0.  , s2_ , a23_],
                  [0.  , 0.  , s3_ ]])
    O = np.array([O0, O1, O2])

    phi = np.degrees( np.arctan2(  A[0,1],   A[1,1] ) )   # true v12
    rho = np.degrees( np.arctan2(  A[0,2],   A[2,2] ) )   # true v13
    lam = np.degrees( np.arctan2(  A[1,2],   A[2,2] ) )   # true v23
    angles = np.array([phi, rho, lam])
    
    return {
        "A":      A,
        "O":      O,
        "angles": angles,
        "gains":  np.array([s1_, s2_, s3_])
    }
