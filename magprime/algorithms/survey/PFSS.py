# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  pfss.py                                                     ║
# ║  Package      :  magprime                                                    ║
# ║  Author       :  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  NASA Goddard Space Flight Center — Greenbelt, MD 20771      ║
# ║  Created      :  2025-05-21                                                  ║
# ║  Last Updated :  2025-05-22                                                  ║
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
import warnings
from numba import jit, prange


@jit(nopython=True, cache=True)
def _hard_threshold_inplace(M, zeta):
    """In-place hard thresholding with Numba acceleration."""
    flat = M.flat
    for i in range(len(flat)):
        if abs(flat[i]) < zeta:
            flat[i] = 0.0


# -----------------------------------------------------------------------------
# OptimizedPFSS class definition
# -----------------------------------------------------------------------------
class OptimizedPFSS:
    def __init__(self, use_gpu=False):
        """
        Initialize optimized PFSS solver.
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration (requires CuPy)
        """
        self.use_gpu = use_gpu
        if use_gpu:
            try:
                import cupy as cp
                self.cp = cp
                self.gpu_available = True
            except ImportError:
                warnings.warn("CuPy not available, falling back to CPU")
                self.gpu_available = False
        else:
            self.gpu_available = False

        # Cache for FFT plans and workspace
        self._fft_cache = {}
        self._workspace_cache = {}

    def _get_array_module(self, X):
        """Get appropriate array module (numpy or cupy)."""
        if self.gpu_available and hasattr(X, '__cuda_array_interface__'):
            return self.cp
        return np

    def _cached_fft_handles(self, X):
        """Build FFT handles with caching for repeated use."""
        P, Q = X.shape
        cache_key = (P, Q, type(X).__name__)
        if cache_key in self._fft_cache:
            return self._fft_cache[cache_key]

        xp = self._get_array_module(X)
        K = (P + 1) // 2
        L = P - K + 1
        Kh = (Q + 1) // 2
        Lh = Q - Kh + 1
        s0, s1 = P + L - 1, Q + Lh - 1

        workspace = {
            'Xpad': xp.zeros((s0, s1), dtype=X.dtype),
            'Wpad_buffer': None,
            'conv_buffer': None
        }

        def matvec(Omega):
            ncols = Omega.shape[1]
            if workspace['Wpad_buffer'] is None or workspace['Wpad_buffer'].shape[2] < ncols:
                workspace['Wpad_buffer'] = xp.zeros((s0, s1, ncols), dtype=X.dtype)
                workspace['conv_buffer'] = xp.zeros((s0, s1, ncols), dtype=xp.complex128)
            Wpad = workspace['Wpad_buffer'][:, :, :ncols]
            Wpad.fill(0)
            W3 = Omega.reshape(L, Lh, ncols, order='F')
            Wpad[:L, :Lh, :] = W3[::-1, ::-1, :]
            workspace['Xpad'][:P, :Q] = X
            X_fft = xp.fft.fft2(workspace['Xpad'])
            W_fft = xp.fft.fft2(Wpad, axes=(0, 1))
            conv_out = xp.fft.ifft2(X_fft[:, :, None] * W_fft, axes=(0, 1)).real
            valid_block = conv_out[L-1:P, Lh-1:Q, :]
            return valid_block.reshape(K * Kh, ncols, order='F')

        def rmatvec(W):
            ncols = W.shape[1]
            if workspace['Wpad_buffer'] is None or workspace['Wpad_buffer'].shape[2] < ncols:
                workspace['Wpad_buffer'] = xp.zeros((s0, s1, ncols), dtype=X.dtype)
            Wpad = workspace['Wpad_buffer'][:, :, :ncols]
            Wpad.fill(0)
            W3 = W.reshape(K, Kh, ncols, order='F')
            Wpad[:K, :Kh, :] = W3[::-1, ::-1, :]
            workspace['Xpad'][:P, :Q] = X
            X_fft = xp.fft.fft2(workspace['Xpad'])
            W_fft = xp.fft.fft2(Wpad, axes=(0, 1))
            conv_out = xp.fft.ifft2(X_fft[:, :, None] * W_fft, axes=(0, 1)).real
            valid_block = conv_out[K-1:P, Kh-1:Q, :]
            return valid_block.reshape(L * Lh, ncols, order='F')

        handles = (matvec, rmatvec, (K, L, Kh, Lh, s0, s1), workspace)
        self._fft_cache[cache_key] = handles
        return handles

    def adaptive_rank_selection(self, X, max_rank=None):
        """
        Automatically determine optimal rank using spectral gap analysis.
        """
        if max_rank is None:
            max_rank = min(X.shape) // 4
        matvec, rmatvec, dims, _ = self._cached_fft_handles(X)
        U, s_vals, _ = self._randomized_svd(matvec, rmatvec, dims, rank=max_rank)
        gaps = np.diff(s_vals)
        gap_ratios = gaps / s_vals[:-1]
        optimal_rank = np.argmax(gap_ratios) + 1
        return min(optimal_rank, max_rank)

    def fast_rsvd_optimized(self, X, rank=10, p=None, q=None, power_scheme='auto'):
        """Optimized randomized SVD with acceleration strategies."""
        matvec, rmatvec, dims, _ = self._cached_fft_handles(X)
        return self._randomized_svd_fft(matvec, rmatvec, dims, rank, p, q, power_scheme)


    def _randomized_svd_fft(self, matvec, rmatvec, dims, rank=10, p=None, q=None, power_scheme='auto'):
        """FFT-based randomized SVD for block-Hankel matrices."""
        K, L, Kh, Lh, *_ = dims
        m, n = K * Kh, L * Lh
        if p is None:
            p = min(10, max(5, rank // 2))
        if q is None:
            q = 2 if (power_scheme == 'auto' and rank < min(m, n) // 4) else 1
        ell = min(rank + p, min(m, n))
        Omega = self._generate_structured_random_matrix(n, ell)
        Y = matvec(Omega)
        if power_scheme == 'adaptive':
            Y = self._adaptive_power_iterations(Y, matvec, rmatvec, q)
        else:
            for _ in range(q):
                Y = matvec(rmatvec(Y))
        Q_mat = self._stable_qr(Y)
        B = rmatvec(Q_mat).T
        if B.shape[0] > ell:
            B = B[:ell, :]
        Ub, s_vals, Vt = svd(B, full_matrices=False)
        U_final = Q_mat @ Ub[:, :rank]
        return U_final, s_vals[:rank], Vt[:rank, :]

    def _randomized_svd_direct(self, X, rank=10, p=None, q=None, power_scheme='auto'):
        """Direct randomized SVD for dense matrices."""
        m, n = X.shape
        if p is None:
            p = min(10, max(5, rank // 2))
        if q is None:
            q = 2 if power_scheme != 'fixed' else 1
        ell = min(rank + p, min(m, n))
        Omega = self._generate_structured_random_matrix(n, ell)
        Y = X @ Omega
        if power_scheme == 'adaptive':
            prev_norm = np.linalg.norm(Y, 'fro')
            for i in range(q):
                Y = X @ (X.T @ Y)
                curr_norm = np.linalg.norm(Y, 'fro')
                if i > 0 and curr_norm / prev_norm < 1.05:
                    break
                prev_norm = curr_norm
        else:
            for _ in range(q):
                Y = X @ (X.T @ Y)
        Q_mat = self._stable_qr(Y)
        B = Q_mat.T @ X
        Ub, s_vals, Vt = svd(B, full_matrices=False)
        U_final = Q_mat @ Ub[:, :rank]
        return U_final, s_vals[:rank], Vt[:rank, :]

    def _generate_structured_random_matrix(self, n, ell, matrix_type='gaussian'):
        """Generate structured random matrices for better numerical properties."""
        if matrix_type == 'gaussian':
            return np.random.randn(n, ell).astype(np.float32)
        elif matrix_type == 'rademacher':
            return np.random.choice([-1, 1], size=(n, ell)).astype(np.float32)
        elif matrix_type == 'srft':
            D = np.random.choice([-1, 1], size=n)
            indices = np.random.choice(n, size=ell, replace=False)
            Omega = np.zeros((n, ell), dtype=np.complex64)
            for j, idx in enumerate(indices):
                Omega[:, j] = D
                Omega[:, j] = np.fft.fft(Omega[:, j])
                Omega[idx, j] *= np.sqrt(n / ell)
            return Omega.real.astype(np.float32)
        else:
            return np.random.randn(n, ell).astype(np.float32)

    def _adaptive_power_iterations(self, Y, matvec, rmatvec, max_iters):
        """Adaptive power iterations with convergence monitoring."""
        prev_norm = np.linalg.norm(Y, 'fro')
        for i in range(max_iters):
            Y_new = matvec(rmatvec(Y))
            curr_norm = np.linalg.norm(Y_new, 'fro')
            if i > 0 and curr_norm / prev_norm < 1.02:
                break
            Y = Y_new
            prev_norm = curr_norm
        return Y

    def _stable_qr(self, Y, reorthogonalize=True):
        """Stable QR decomposition with optional reorthogonalization."""
        Q_mat, R = qr(Y, mode='economic')
        if reorthogonalize and Q_mat.shape[1] > 10:
            Q_mat, _ = qr(Q_mat, mode='economic')
        return Q_mat

    def _randomized_svd(self, matvec, rmatvec, dims, rank=10, p=5, q=2):
        """Legacy randomized SVD method (backward compatibility)."""
        return self._randomized_svd_fft(matvec, rmatvec, dims,
                                        rank=rank, p=p, q=q,
                                        power_scheme='auto')

    def fast_inverse_block_hankel_vectorized(self, U, s, Vt, P, Q):
        """Optimized inverse block Hankel with vectorized operations."""
        xp = self._get_array_module(U)
        block_rows = (P + 1) // 2
        block_cols = P - block_rows + 1
        grid_cols = (Q + 1) // 2
        grid_rows = Q - grid_cols + 1
        if not hasattr(self, '_counts_cache') or self._counts_cache[0] != (P, Q):
            counts = fftconvolve(
                np.ones((block_rows, grid_cols)),
                np.ones((block_cols, grid_rows)),
                mode="full"
            )
            self._counts_cache = ((P, Q), counts)
        else:
            counts = self._counts_cache[1]
        U_reshaped = U.reshape(block_rows, grid_cols, -1, order='F')
        V_reshaped = Vt.reshape(-1, block_cols, grid_rows, order='F')
        X_acc = xp.zeros((P, Q))
        for k in range(len(s)):
            conv_result = fftconvolve(U_reshaped[:,:,k], V_reshaped[k], mode="full")
            X_acc += s[k] * conv_result
        return X_acc / counts

    def pfss_optimized(self, X, r_max=None, beta=0.8, max_iter=10, eps=1e-4,
                      adaptive_threshold=True, early_stopping=True, verbose=False):
        """Optimized PFSS with multiple acceleration techniques."""
        P, Q = X.shape
        xp = self._get_array_module(X)
        if r_max is None:
            r_max = self.adaptive_rank_selection(X)
            if verbose:
                print(f"Auto-selected rank: {r_max}")
        matvec, rmatvec, dims, workspace = self._cached_fft_handles(X)
        U_init, s_init, _ = self._randomized_svd(matvec, rmatvec, dims, rank=min(5, r_max))
        zeta0 = beta * s_init[0] if len(s_init)>0 else beta * np.std(X)
        X_low = xp.zeros_like(X)
        X_sparse = X.copy()
        _hard_threshold_inplace(X_sparse, zeta0)
        prev_residual_norm = np.inf
        stagnation_count = 0
        for k in range(1, r_max+1):
            if verbose:
                print(f"\n-- Target rank k = {k}")
            converged = False
            for t in range(max_iter):
                residual = X - X_low
                matvec_res, rmatvec_res, _, _ = self._cached_fft_handles(residual)
                U, s_vals, Vt = self._randomized_svd(matvec_res, rmatvec_res, dims, rank=min(k+2, r_max+1))
                if len(s_vals) <= k:
                    break
                if adaptive_threshold and k < len(s_vals):
                    gap_ratio = (s_vals[k-1] - s_vals[k]) / s_vals[k-1]
                    zeta = beta * s_vals[k] * (1 + gap_ratio)
                else:
                    zeta = beta * s_vals[k] if k<len(s_vals) else beta*s_vals[-1]
                X_low_new = self.fast_inverse_block_hankel_vectorized(U[:,:k], s_vals[:k], Vt[:k,:], P, Q)
                X_sparse_new = X - X_low_new
                _hard_threshold_inplace(X_sparse_new, zeta)
                delta = np.linalg.norm(X_low_new - X_low)
                residual_norm = np.linalg.norm(X - X_low_new - X_sparse_new)
                if verbose:
                    print(f"  Iter {t}: ΔX_low = {delta:.2e}, residual = {residual_norm:.2e}")
                if early_stopping and delta<eps:
                    converged = True
                    break
                if early_stopping and abs(residual_norm-prev_residual_norm)/prev_residual_norm<0.01:
                    stagnation_count+=1
                    if stagnation_count>=3:
                        if verbose:
                            print("  Early stopping due to stagnation")
                        break
                else:
                    stagnation_count=0
                prev_residual_norm = residual_norm
                X_low, X_sparse = X_low_new, X_sparse_new
                if converged:
                    break
            if early_stopping and k>2:
                current_error = np.linalg.norm(X - X_low - X_sparse)
                if current_error/np.linalg.norm(X)<0.01:
                    if verbose:
                        print(f"Early termination at rank {k}, relative error: {current_error/np.linalg.norm(X):.2e}")
                    break
        X_sparse = X - X_low
        return X_low, X_sparse

    def ssa_optimized(self, X, r_max=None, sort_components=True, batch_reconstruct=True):
        """Optimized Singular Spectrum Analysis with vectorized reconstruction."""
        P, Q = X.shape
        xp = self._get_array_module(X)
        if r_max is None:
            r_max = self.adaptive_rank_selection(X, max_rank=min(P,Q)//3)
        matvec, rmatvec, dims, _ = self._cached_fft_handles(X)
        U, s, Vt = self._randomized_svd(matvec, rmatvec, dims, rank=r_max)
        if sort_components:
            idx = np.argsort(-s)
            s, U, Vt = s[idx], U[:,idx], Vt[idx,:]
        if batch_reconstruct:
            comps = self._batch_reconstruct_components(U, s, Vt, P, Q)
        else:
            comps = self._individual_reconstruct_components(U, s, Vt, P, Q)
        return comps

    def _batch_reconstruct_components(self, U, s, Vt, P, Q):
        """Batch reconstruction of all SSA components."""
        xp = self._get_array_module(U)
        r = len(s)
        K, Kh = (P + 1)//2, (Q + 1)//2
        L, Lh = P - K + 1, Q - Kh + 1
        cache_key = (P,Q,'ssa_counts')
        if cache_key not in self._workspace_cache:
            counts = fftconvolve(np.ones((K,Kh)), np.ones((L,Lh)), mode='full')
            self._workspace_cache[cache_key] = counts
        else:
            counts = self._workspace_cache[cache_key]
        U_batch = U.reshape(K,Kh,r,order='F')
        V_batch = Vt.reshape(r,L,Lh,order='F')
        comps = xp.zeros((r,P,Q),dtype=U.dtype)
        chunk_size = min(8, r)
        for i in range(0,r,chunk_size):
            end_idx = min(i+chunk_size, r)
            for k in range(i,end_idx):
                conv_result = fftconvolve(U_batch[:,:,k], V_batch[k], mode='full')
                comps[k] = s[k]*conv_result/ counts
        return comps

    def _individual_reconstruct_components(self, U, s, Vt, P, Q):
        """Individual component reconstruction (fallback)."""
        xp = self._get_array_module(U)
        r = len(s)
        K, Kh = (P + 1)//2, (Q + 1)//2
        L, Lh = P - K + 1, Q - Kh + 1
        cache_key = (P,Q,'ssa_counts')
        if cache_key not in self._workspace_cache:
            counts = fftconvolve(np.ones((K,Kh)), np.ones((L,Lh)), mode='full')
            self._workspace_cache[cache_key] = counts
        else:
            counts = self._workspace_cache[cache_key]
        comps = xp.zeros((r,P,Q),dtype=U.dtype)
        for k in range(r):
            Uk = U[:,k].reshape(K,Kh,order='F')
            Vk = Vt[k].reshape(L,Lh,order='F')
            comps[k] = s[k]*fftconvolve(Uk, Vk, mode='full')/ counts
        return comps

    def ssa_from_svd_optimized(self, U, s, Vt, shape, batch_reconstruct=True):
        """Optimized SSA reconstruction from existing SVD."""
        P,Q = shape
        if batch_reconstruct:
            return self._batch_reconstruct_components(U, s, Vt, P, Q)
        else:
            return self._individual_reconstruct_components(U, s, Vt, P, Q)

# -----------------------------------------------------------------------------
# Standalone functions for backward compatibility
# -----------------------------------------------------------------------------
def fast_rsvd(X, rank=10, p=1, q=1, power_scheme='auto', use_gpu=False):
    solver = OptimizedPFSS(use_gpu=use_gpu)
    return solver.fast_rsvd_optimized(X, rank, p, q, power_scheme)

def pfss(X, r_max=None, beta=0.8, max_iter=10, eps=1e-4, verbose=False, use_gpu=False):
    solver = OptimizedPFSS(use_gpu=use_gpu)
    return solver.pfss_optimized(X, r_max, beta, max_iter, eps, verbose=verbose)

def ssa(X, r_max=None, sort_components=True, use_gpu=False):
    solver = OptimizedPFSS(use_gpu=use_gpu)
    return solver.ssa_optimized(X, r_max=r_max, sort_components=sort_components)

def ssa_from_svd(U, s, Vt, shape, use_gpu=False):
    solver = OptimizedPFSS(use_gpu=use_gpu)
    return solver.ssa_from_svd_optimized(U, s, Vt, shape)
