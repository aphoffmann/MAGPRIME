# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  pfss.py                                                     ║
# ║  Package      :  magprime                                                    ║
# ║  Author       :  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-05-21                                                  ║
# ║  Last Updated :  2025-05-21                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  : Non-convex low-rank + sparse decomposition via PFSS          ║
# ║                 ─────────────────────────────────────────────────────────    ║
# ║                 • fast_rsvd: randomized SVD of implicit block-Hankel         ║
# ║                   trajectory matrix using FFT-based matmul and power iters   ║
# ║                 • fast_inverse_block_hankel: reconstructs low-rank           ║
# ║                   approximation by convolving rank-1 factors over Hankel     ║
# ║                   blocks and normalizing by overlap counts                   ║
# ║                 • pfss: Separates spatial signals by sparsity                ║
# ║                 • ssa: performs Singular Spectrum Analysis by extracting     ║
# ║                   elementary components via fast_rsvd and inverse Hankel     ║
# ║                   projection                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import numpy as np
from scipy.linalg import svd, qr
from scipy.signal import fftconvolve

def form_block_hankel(X):
    P, Q = X.shape
    K  = (P + 1) // 2
    L  = P - K + 1
    Kh = (Q + 1) // 2
    Lh = Q - Kh + 1

    T = np.zeros((K * Kh, L * Lh))
    for br in range(Kh):
        for bc in range(Lh):
            j = br + bc
            if j < Q:
                hankel_block = np.array(
                    [[X[r + c, j] if r + c < P else 0
                      for c in range(L)] for r in range(K)]
                )
                T[br * K:(br + 1) * K, bc * L:(bc + 1) * L] = hankel_block
    return T

def inverse_block_hankel(T, P, Q):
    K = (P + 1) // 2
    L = P - K + 1
    Kh = (Q + 1) // 2
    Lh = Q - Kh + 1
    X_acc = np.zeros((P, Q), dtype=T.dtype)
    counts = np.zeros((P, Q), dtype=np.int32)
    for br in range(Kh):
        for bc in range(Lh):
            j = br + bc
            if j < Q:
                blk = T[br*K:(br+1)*K, bc*L:(bc+1)*L]
                for r in range(K):
                    for c in range(L):
                        i = r + c
                        if i < P:
                            X_acc[i, j] += blk[r, c]
                            counts[i, j] += 1
    X_hat = np.divide(X_acc, counts, where=counts > 0)
    return X_hat

def fast_inverse_block_hankel(U, s, Vt, P, Q):
    """
    Reconstruct X from the factorized T_r = U · diag(s) · Vt
    where T_r is the low-rank block-Hankel matrix of X.
    """
    rank = len(s)

    # Determine block-Hankel dimensions
    block_rows = (P + 1) // 2
    block_cols = P - block_rows + 1
    grid_cols  = (Q + 1) // 2
    grid_rows  = Q - grid_cols + 1

    # Precompute overlap counts for normalization
    counts = fftconvolve(
        np.ones((block_rows, grid_cols)),
        np.ones((block_cols, grid_rows)),
        mode="full"
    )

    # Accumulate each rank-one contribution
    X_acc = np.zeros((P, Q))
    for k in range(rank):
        U_k = U[:, k].reshape(block_rows, grid_cols, order='F')
        V_k = Vt[k, :].reshape(block_cols, grid_rows, order='F')
        X_acc += s[k] * fftconvolve(U_k, V_k, mode="full")

    return X_acc / counts

# === build_fft_handles ===
def build_fft_handles(X):
    """
    Construct two callables:
      matvec(Omega)  = T @ Omega
      rmatvec(W)     = T.T @ W
    for T = block-Hankel matrix of X.
    """
    P, Q = X.shape

    # Block sizes
    K   = (P + 1) // 2
    L   = P - K + 1
    Kh  = (Q + 1) // 2
    Lh  = Q - Kh + 1

    # Size for linear conv
    s0 = P + L - 1
    s1 = Q + Lh - 1

    # FFT of padded X (zero-pad once)
    Xpad     = np.zeros((s0, s1))
    Xpad[:P, :Q] = X
    X_fft    = np.fft.fft2(Xpad)

    def _corr_fft(kernel_fft):
        """Helper: real IFFT of X_fft * kernel_fft → full linear correlation."""
        return np.fft.ifft2(X_fft[:, :, None] * kernel_fft, axes=(0,1)).real

    # ----- forward multiply: T @ Omega -----
    def matvec(Omega):
        _, ncols = Omega.shape
        W3       = Omega.reshape(L, Lh, ncols, order='F')
        W_flip   = W3[::-1, ::-1, :]  # flip for correlation

        Wpad     = np.zeros((s0, s1, ncols))
        Wpad[:L, :Lh, :] = W_flip
        conv_out = _corr_fft(np.fft.fft2(Wpad, axes=(0,1)))

        valid_block = conv_out[L-1:P, Lh-1:Q, :]
        return valid_block.reshape(K * Kh, ncols, order='F')

    # ----- adjoint multiply: T.T @ W -----
    def rmatvec(W):
        _, ncols = W.shape
        W3       = W.reshape(K, Kh, ncols, order='F')
        W_flip   = W3[::-1, ::-1, :]

        Wpad     = np.zeros((s0, s1, ncols))
        Wpad[:K, :Kh, :] = W_flip
        conv_out = _corr_fft(np.fft.fft2(Wpad, axes=(0,1)))

        valid_block = conv_out[K-1:P, Kh-1:Q, :]
        return valid_block.reshape(L * Lh, ncols, order='F')

    dims = (K, L, Kh, Lh, s0, s1)
    return matvec, rmatvec, dims


# === fast_rsvd ===
def fast_rsvd(X, rank=10, p=1, q=1):
    """
    Randomized SVD of the block-Hankel matrix of X using FFT handles.
    Returns U, singular_values, Vt with specified rank.
    """
    matmul, rmatmul, dims = build_fft_handles(X)
    K, L, Kh, Lh, *_ = dims
    m, ncols = K * Kh, L * Lh
    ell = rank + p

    # Stage A: sample range of T
    Omega = np.random.randn(ncols, ell)
    Y     = matmul(Omega)
    for _ in range(q):
        Y = matmul(rmatmul(Y))

    # Stage B: orthonormal basis
    Q_mat, _ = qr(Y, mode='economic')

    # Stage C: small SVD
    B = rmatmul(Q_mat).T[:ell, :]
    Ub, s_vals, Vt = svd(B, full_matrices=False)

    # Form final U
    U_final = Q_mat @ Ub[:, :rank]
    return U_final, s_vals[:rank], Vt[:rank, :]

def hard_threshold(M, zeta):
    """
    Zero out entries with absolute value below zeta_value.
    """
    out = M.copy()
    out[np.abs(out) < zeta] = 0.0
    return out

def pfss(X, r_max, beta=0.8, max_iter=10, eps=1e-4, verbose=False):
    """
    Perform a non-convex decomposition of X into a low-rank part and a sparse part.

    This uses an alternating scheme: at each target rank k (up to r_max), it
    computes a truncated RSVD to update the low-rank approximation, then
    applies a hard threshold (scaled by beta) to the residual to extract sparse
    outliers. Iteration at each k continues until the low-rank update changes
    by less than eps or max_iter is reached.

    Parameters
    ----------
    X : ndarray, shape (P, Q)
        Input data matrix to decompose.
    r_max : int
        Maximum rank of the low-rank component.
    beta : float, default=0.8
        Scaling factor for the singular-value‐based threshold used in sparse extraction.
    max_iter : int, default=10
        Maximum number of inner refinement iterations at each rank.
    eps : float, default=1e-4
        Convergence tolerance on the change in the low-rank component.
    verbose : bool, default=False
        If True, print iteration diagnostics.

    Returns
    -------
    X_lowrank : ndarray
        Low-rank approximation of X.
    X_sparse : ndarray
        Sparse residual (X − X_lowrank), containing thresholded “outliers.”
    """
    P, Q = X.shape

    # Initial rank-1 threshold
    _, s1, _ = fast_rsvd(X, rank=1)
    zeta0    = beta * s1[0]
    X_low    = np.zeros_like(X)
    X_sparse = hard_threshold(X, zeta0)

    if verbose:
        print(f"Initial threshold zeta0 = {zeta0:.3e}")

    # Outer loop over target rank
    for k in range(1, r_max + 1):
        if verbose:
            print(f"\n-- Target rank k = {k}")

        for t in range(max_iter):
            U, s_vals, Vt = fast_rsvd(X - X_low, rank=k+1)

            # Adaptive threshold
            zeta = beta * (s_vals[k] + 0.5 * s_vals[k-1])

            # Best rank-k approximation
            X_low_new = fast_inverse_block_hankel(U[:, :k], s_vals[:k], Vt[:k, :], P, Q)
            X_sparse_new = hard_threshold(X - X_low_new, zeta)

            # Convergence check
            delta = np.linalg.norm(X_low_new - X_low)
            if verbose:
                print(f"  Iter {t}: ΔX_low norm = {delta:.2e}, zeta = {zeta:.3e}")

            X_low, X_sparse = X_low_new, X_sparse_new
            if delta < eps:
                break

    # Final sparse component
    X_sparse = X - X_low
    return X_low, X_sparse

def ssa(X, r_max):
    "Take SVD"
    U, s, Vt = fast_rsvd(X, rank=r_max) 

    "Sort by descending singular Values"
    idx      = np.argsort(-s)
    s        = s[idx]
    U        = U[:, idx]
    Vt       = Vt[idx, :]

    "Dimensions & convolution weights for inverse projection"
    P, Q  = X.shape
    K, Kh = (P + 1) // 2, (Q + 1) // 2
    L, Lh = P - K + 1,   Q - Kh + 1
    counts = fftconvolve(np.ones((K, Kh)),
                         np.ones((L, Lh)), mode="full")

    "Helper: reconstruct one elementary component"
    def Xi(i):
        Ui = U[:, i].reshape(K, Kh, order='F')
        Vi = Vt[i, :].reshape(L, Lh, order='F')
        return s[i] * fftconvolve(Ui, Vi, mode='full') / counts

    "stack all r components"
    comps = np.empty((r_max, P, Q), dtype=X.dtype)
    for k in range(r_max):
        comps[k] = Xi(k)

    return comps

def ssa_from_svd(U, s, Vt, shape):
    P, Q   = shape
    K, Kh  = (P + 1)//2, (Q + 1)//2
    L, Lh  = P - K + 1,  Q - Kh + 1
    counts = fftconvolve(np.ones((K,Kh)), np.ones((L,Lh)), mode='full')

    r      = len(s)
    comps  = np.empty((r, P, Q), dtype=U.dtype)
    for k in range(r):
        Uk = U[:,k].reshape(K,Kh, order='F')
        Vk = Vt[k].reshape(L,Lh, order='F')
        comps[k] = s[k] * fftconvolve(Uk, Vk, mode='full') / counts
    return comps







