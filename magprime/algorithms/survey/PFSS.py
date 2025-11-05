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
from numba import jit
from functools import cache
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance   import squareform

@cache
def _build_fft_handles(shape: tuple[int, int], type_name: str):
    """
    Build and cache FFT handles for a given array shape and type.
    Returns (matvec, rmatvec, dims, workspace).
    """
    P, Q = shape
    K = (P + 1) // 2
    L = P - K + 1
    Kh = (Q + 1) // 2
    Lh = Q - Kh + 1
    s0, s1 = P + L - 1, Q + Lh - 1

    # Workspace buffers for FFT padding and convolution
    workspace = {
        'pad': np.zeros((s0, s1), dtype=np.float64),
        'Wpad': None
    }

    def matvec(Omega: np.ndarray) -> np.ndarray:
        ncols = Omega.shape[1]
        # Allocate or resize Wpad
        if workspace['Wpad'] is None or workspace['Wpad'].shape[2] < ncols:
            workspace['Wpad'] = np.zeros((s0, s1, ncols), dtype=Omega.dtype)
        Wpad = workspace['Wpad'][:, :, :ncols]
        Wpad.fill(0)
        # Fill reversed Hankel blocks
        W3 = Omega.reshape(L, Lh, ncols, order='F')
        Wpad[:L, :Lh, :] = W3[::-1, ::-1, :]
        # FFT of original data
        workspace['pad'][:P, :Q] = X_global  # X_global is set by caller context
        X_fft = np.fft.fft2(workspace['pad'])
        # FFT of blocks and convolution
        W_fft = np.fft.fft2(Wpad, axes=(0, 1))
        conv = np.fft.ifft2(X_fft[:, :, None] * W_fft, axes=(0, 1)).real
        block = conv[L - 1:P, Lh - 1:Q, :]
        return block.reshape(K * Kh, ncols, order='F')

    def rmatvec(W: np.ndarray) -> np.ndarray:
        ncols = W.shape[1]
        if workspace['Wpad'] is None or workspace['Wpad'].shape[2] < ncols:
            workspace['Wpad'] = np.zeros((s0, s1, ncols), dtype=W.dtype)
        Wpad = workspace['Wpad'][:, :, :ncols]
        Wpad.fill(0)
        W3 = W.reshape(K, Kh, ncols, order='F')
        Wpad[:K, :Kh, :] = W3[::-1, ::-1, :]
        workspace['pad'][:P, :Q] = X_global
        X_fft = np.fft.fft2(workspace['pad'])
        W_fft = np.fft.fft2(Wpad, axes=(0, 1))
        conv = np.fft.ifft2(X_fft[:, :, None] * W_fft, axes=(0, 1)).real
        block = conv[K - 1:P, Kh - 1:Q, :]
        return block.reshape(L * Lh, ncols, order='F')

    return matvec, rmatvec, (K, L, Kh, Lh, s0, s1), workspace

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
class PFSS:
    def __init__(self):
        """
        Initialize optimized PFSS solver.
        Parameters
        ----------
        use_gpu : bool
            Whether to use GPU acceleration (requires CuPy)
        """
        # Cache for FFT plans and workspace
        self._fft_cache = {}
        self._workspace_cache = {}
        self.s = None

    def _cached_fft_handles(self, X: np.ndarray):
        """
        Retrieve FFT handles for X via functools.cache.
        """
        # Set global for closures
        global X_global
        X_global = X
        return _build_fft_handles(X.shape, type(X).__name__)

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
        X_acc = np.zeros((P, Q))
        for k in range(len(s)):
            conv_result = fftconvolve(U_reshaped[:,:,k], V_reshaped[k], mode="full")
            X_acc += s[k] * conv_result
        return X_acc / counts

    def pfss_optimized(
        self,
        X: np.ndarray,
        r_max: int | None = None,
        beta: float = 0.8,
        max_iter: int = 10,
        eps: float = 1e-4,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Low-rank + sparse decomposition via alternating Hankel low-rank (SSA)
        and hard thresholding of the residual with a median-based tau.
        """
        P, Q = X.shape
        normX = np.linalg.norm(X)

        # 1) choose rank
        if r_max is None:
            r_max = self.adaptive_rank_selection(X)
            if verbose:
                print(f"Auto-selected rank = {r_max}")

        # 2) initial low-rank from the full X
        matvec, rmatvec, dims, _ = self._cached_fft_handles(X)
        U, s_vals, Vt = self._randomized_svd(matvec, rmatvec, dims, rank=r_max)
        X_low = self.fast_inverse_block_hankel_vectorized(U, s_vals, Vt, P, Q)

        # 3) initial sparse = hard-threshold(residual)
        residual = X - X_low
        tau = beta * np.median(np.abs(residual))
        X_sparse = residual.copy()
        _hard_threshold_inplace(X_sparse, tau)

        prev_low    = X_low
        prev_sparse = X_sparse

        # 4) alternate
        for it in range(1, max_iter+1):
            if verbose:
                print(f"-- PFSS iter {it}/{max_iter}")

            # low-rank step on X − sparse
            Y = X - X_sparse
            matvec, rmatvec, dims, _ = self._cached_fft_handles(Y)
            U, s_vals, Vt = self._randomized_svd(matvec, rmatvec, dims, rank=r_max)
            self.s = s_vals
            X_low = self.fast_inverse_block_hankel_vectorized(U, s_vals, Vt, P, Q)

            # sparse step on X − low
            residual = X - X_low
            tau = beta * np.median(np.abs(residual))
            X_sparse = residual.copy()
            _hard_threshold_inplace(X_sparse, tau)

            # check convergence in relative Frobenius norm
            low_diff    = np.linalg.norm(X_low - prev_low)    / normX
            sparse_diff = np.linalg.norm(X_sparse - prev_sparse) / normX

            if verbose:
                print(f"  low_diff = {low_diff:.2e}, sparse_diff = {sparse_diff:.2e}")

            if low_diff < eps and sparse_diff < eps:
                if verbose:
                    print("Converged, stopping early")
                break

            prev_low    = X_low
            prev_sparse = X_sparse

        return X_low, X_sparse

    def ssa_optimized(self, X, r_max=None, sort_components=True, batch_reconstruct=True):
        """Optimized Singular Spectrum Analysis with vectorized reconstruction."""
        P, Q = X.shape
        matvec, rmatvec, dims, _ = self._cached_fft_handles(X)
        K, L, Kh, Lh, *_ = dims
        m, n = K * Kh, L * Lh

        if r_max is None:
            r_max = self.adaptive_rank_selection(X, max_rank=min(P,Q)//3)
        elif r_max < 0:
            # full‐rank reconstruction
            r_max = min(m, n)

        U, s, Vt = self._randomized_svd(matvec, rmatvec, dims, rank=r_max)
        self.s = s
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
        comps = np.zeros((r,P,Q),dtype=U.dtype)
        chunk_size = min(8, r)
        for i in range(0,r,chunk_size):
            end_idx = min(i+chunk_size, r)
            for k in range(i,end_idx):
                conv_result = fftconvolve(U_batch[:,:,k], V_batch[k], mode='full')
                comps[k] = s[k]*conv_result/ counts
        return comps

    def _individual_reconstruct_components(self, U, s, Vt, P, Q):
        """Individual component reconstruction (fallback)."""
        r = len(s)
        K, Kh = (P + 1)//2, (Q + 1)//2
        L, Lh = P - K + 1, Q - Kh + 1
        cache_key = (P,Q,'ssa_counts')
        if cache_key not in self._workspace_cache:
            counts = fftconvolve(np.ones((K,Kh)), np.ones((L,Lh)), mode='full')
            self._workspace_cache[cache_key] = counts
        else:
            counts = self._workspace_cache[cache_key]
        comps = np.zeros((r,P,Q),dtype=U.dtype)
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

    def group_components(
        self,
        comps: np.ndarray,
        n_groups:    int    = None,
        threshold:  float   = None,
        linkage_method: str = 'average'
        ) -> tuple[np.ndarray, list[list[int]]]:
            """
            Automatic grouping of 2D-SSA components via weighted-correlation + hierarchical clustering.

            Parameters
            ----------
            comps : array_like, shape (r, P, Q)
                The r SSA elementary reconstructions.
            n_groups : int, optional
                If set, cut the dendrogram into this many clusters.
            threshold : float, optional
                If set, cut the dendrogram at this dissimilarity threshold.
            linkage_method : str
                One of {'single','complete','average','ward',…}.

            Returns
            -------
            labels : ndarray, shape (r,)
                Cluster label (1…k) for each component.
            groups : list of lists
                groups[i] is the list of component-indices in cluster i+1.
            """
            r, P, Q = comps.shape

            # 1) flatten each component to a vector
            flat = comps.reshape(r, -1)

            # 2) weighted-correlation matrix w_ij
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            corr  = (flat @ flat.T) / (norms @ norms.T)
            wcorr = corr  # in [-1..1]

            # 3) dissimilarity = 1 - |w|
            diss = 1 - np.abs(wcorr)

            # 4) hierarchical clustering
            condensed = squareform(diss, checks=False)
            Z = linkage(condensed, method=linkage_method)

            # 5) decide where to cut
            if n_groups is not None:
                labels = fcluster(Z, t=n_groups,    criterion='maxclust')
            elif threshold is not None:
                labels = fcluster(Z, t=threshold,   criterion='distance')
            else:
                raise ValueError("must specify either n_groups or threshold")

            # 6) collect the groups
            groups = []
            for k in range(1, labels.max()+1):
                groups.append(list(np.nonzero(labels==k)[0]))

            return labels, groups
    
# -----------------------------------------------------------------------------
# Standalone functions for backward compatibility
# -----------------------------------------------------------------------------
def fast_rsvd(X, rank=10, p=1, q=1, power_scheme='auto'):
    solver = PFSS()
    return solver.fast_rsvd_optimized(X, rank, p, q, power_scheme)

def pfss(X, r_max=None, beta=0.8, max_iter=10, eps=1e-4, verbose=False, ):
    solver = PFSS()
    return solver.pfss_optimized(X, r_max, beta, max_iter, eps, verbose=verbose)

def ssa(X, r_max=None, sort_components=True):
    solver = PFSS()
    return solver.ssa_optimized(X, r_max=r_max, sort_components=sort_components)

def ssa_from_svd(U, s, Vt, shape):
    solver = PFSS()
    return solver.ssa_from_svd_optimized(U, s, Vt, shape)
