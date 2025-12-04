# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              █ █ █ █ █   MAGPRIME Toolkit   █ █ █ █ █                        ║
# ║ ──────────────────────────────────────────────────────────────────────────── ║
# ║  Module       :  DAFGrad.py                                                  ║
# ║  Package      :  magprime                                                    ║
# ║  Author       :  Dr. Ovidiu Dragos Constantinescu                            ║
# ║                  Dr. Alex P. Hoffmann  <alex.p.hoffmann@nasa.gov>            ║
# ║  Affiliation  :  Institut für Geophysik und extraterrestrische Physik        ║
# ║                  NASA Goddard Space Flight Center                            ║
# ║  Created      :  2025-11-21                                                  ║
# ║  Last Updated :  2025-11-22                                                  ║
# ║  Python       :  ≥ 3.10                                                      ║
# ║  License      :  MIT — see LICENSE.txt                                       ║
# ║                                                                              ║
# ║  Description  : Todo                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from nsgt import CQ_NSGT

"Algorithm Parameters"
fs = 1
bpo = 10
ssp_similarity = 0.95
magnitude_percentile = 90.0
ambient_similarity = 0.8

def clean(B, triaxial = True, sens=0):
    """
    Parameters
    ----------
    B : ndarray (n_sensors, axes, n_samples)
        Magnetic field measurements from the sensor array.
    triaxial : bool, optional
        The default is True.
    sens : int, optional
        The sensor number. The default is 0.

    Raises
    ------
    Exception
        If triaxial is False or if order > 3.

    Returns
    -------
    ndarray  (axes, n_samples)
        Corrected measurements corresponding to sensor sens.

    """
    
    if(triaxial == False):
        raise Exception("'triaxial' is set to False. PiCoG only works for triaxial data")
    
    # Take NSGT of B
    length = B.shape[-1]
    bpo = 10
    bins = bpo
    fmax = fs/2
    lowf = 2 * bpo * fs / length
    nsgt = CQ_NSGT(lowf, fmax, bins, fs, length, multichannel=True)

    Bf = []
    for i in range(B.shape[0]):
        Bi = np.array(nsgt.forward(B[i]), dtype=object)
        Bi = np.vstack([np.hstack(Bi[j]) for j in range(Bi.shape[0])])
        Bf.append(Bi)
    Sf = np.array(Bf)

    # Find SSPs
    SSP_mask = is_SSP(Sf, similarity=ssp_similarity)
    MAG_mask = magnitude_filter(Sf, percentile=magnitude_percentile)
    ambient_mask = ambient_filter(Sf, similarity=ambient_similarity)
    mask = SSP_mask & MAG_mask & (~ambient_mask)

    SSP_data = Sf[:, :, mask] 
    print("Number of SSP bins:", SSP_data.shape[2])

    # Estimate disturbance source parameters
    h1_dirs, h2_dirs, k_gain = cluster_gain_and_dir(SSP_data)

    # Clean signals
    B_clean = dafgrad_clean(
        B,
        h1_dirs=h1_dirs,
        h2_dirs=h2_dirs,
        k_gain=k_gain,
    )

    return B_clean[sens]

def sph_to_cart(theta_deg, phi_deg, mag=1.0):
    th = np.deg2rad(theta_deg)
    ph = np.deg2rad(phi_deg)
    return np.array([
        mag*np.sin(th)*np.cos(ph),
        mag*np.sin(th)*np.sin(ph),
        mag*np.cos(th)
    ], float)

def hat(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector cannot be normalized")
    return v / n

def make_skew_frame(dirs):
    """
    Build 3x3 skewed frame:
    first len(dirs) columns are exactly the given unit directions,
    remaining column(s) just complete a basis.
    """
    dirs = [hat(v) for v in dirs]
    m = len(dirs)

    if m == 0:
        return np.eye(3)

    if m == 1:
        v1 = dirs[0]
        cand = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u2 = hat(cand - np.dot(cand, v1) * v1)
        u3 = hat(np.cross(v1, u2))
        return np.column_stack([v1, u2, u3])

    if m == 2:
        v1, v2 = dirs
        # If nearly colinear, tweak v2
        if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
            cand = np.array([1.0, 0.0, 0.0])
            v2 = hat(cand - np.dot(cand, v1) * v1)
        u3 = hat(np.cross(v1, v2))
        return np.column_stack([v1, v2, u3])

    # m >= 3: just take first three directions as they are
    v1, v2, v3 = dirs[:3]
    return np.column_stack([v1, v2, v3])

def is_SSP(Sf, similarity=0.98):
    """
    Sf: complex STFT, shape (n_sensors, 3, N)
    similarity: threshold on |cos(Re, Im)| for SSP detection
    Returns: mask of shape (N,), True where bin is SSP.
    """
    n_sensors, n_axis, N = Sf.shape

    # Flatten sensors and axes into a single "channel" dimension
    X = Sf.reshape(n_sensors * n_axis, N)  # (M, N), M = n_sensors * 3

    A = np.real(X)
    B = np.imag(X)

    # Dot product and norms along channel axis
    dot = np.sum(A * B, axis=0)                       # (F, T)
    normA = np.linalg.norm(A, axis=0)                 # (F, T)
    normB = np.linalg.norm(B, axis=0)                 # (F, T)

    # Cosine similarity, safe divide
    cos_sim = np.divide(
        dot,
        normA * normB,
        out=np.zeros_like(dot),
        where=(normA * normB) != 0
    )

    # SSP if real and imag are (anti-)parallel
    return np.abs(cos_sim) >= similarity

def magnitude_filter(Sf, percentile):
    """
    Sf: (n_sensors, 3,N)
    Returns: mask (N,) where multi-channel magnitude is above threshold.
    """
    n_sensors, n_axis, N = Sf.shape
    X = Sf.reshape(n_sensors * n_axis,N)
    m = np.linalg.norm(X, axis=0)  # (F, T)
    threshold = np.percentile(m, percentile)
    return m >= threshold

def ambient_filter(Sf, similarity=0.98):
    """
    Equal-gain ambient filter using triad magnitudes per sensor.
    """
    n_sensors, n_axis, N = Sf.shape
    mags = np.linalg.norm(Sf, axis=1)  # (n_sensors, F, T)

    ones = np.ones_like(mags)
    dot = np.sum(mags * ones, axis=0)
    normA = np.linalg.norm(mags, axis=0)
    normB = np.linalg.norm(ones, axis=0)

    cos_sim = np.divide(dot,
                        normA * normB,
                        out=np.zeros_like(dot),
                        where=(normA * normB) != 0)

    return cos_sim >= similarity

def remove_phase(v, ref='max'):
    v = np.asarray(v, np.complex128)
    if np.allclose(v, 0):
        return v, 0.0

    if ref == 'max':
        idx = np.argmax(np.abs(v))
        phi_ref = np.angle(v[idx])
    elif ref == 'first':
        phi_ref = np.angle(v[0])
    elif isinstance(ref, (int, float)):
        phi_ref = float(ref)
    else:
        raise ValueError("ref must be 'max', 'first', or a float")

    return v * np.exp(-1j * phi_ref), phi_ref

def remove_global_phase_two_sensor(v1, v2):
    """
    Remove a single global phase shared across both triads.

    v1, v2 : complex (3,)
    Returns:
        v1p, v2p : phase-aligned complex triads
    """
    # Stack components from both sensors, pick the strongest as phase reference
    stacked = np.concatenate([v1, v2])
    idx = np.argmax(np.abs(stacked))
    if np.abs(stacked[idx]) == 0:
        return v1, v2  # degenerate, let caller handle

    phi = np.angle(stacked[idx])
    factor = np.exp(-1j * phi)
    return v1 * factor, v2 * factor

def cluster_gain_and_dir(SSP_data, min_cluster_size=20, n_keep=2):
    """
    Estimate disturbance source directions and gains from SSP bins.

    Parameters
    ----------
    SSP_data : np.ndarray
        Shape (2, 3, n_bins), complex STFT values at selected SSP bins.
    min_cluster_size : int
        Minimum cluster size for HDBSCAN.
    n_keep : int
        Number of largest clusters (sources) to return.

    Returns
    -------
    h1_dirs : (m, 3)
        Unit disturbance directions at sensor 1.
    h2_dirs : (m, 3)
        Unit disturbance directions at sensor 2.
    k_gain : (m,)
        Gain per source, |H2| / |H1|.
    """
    B1 = SSP_data[0].T  # (n_bins, 3)
    B2 = SSP_data[1].T  # (n_bins, 3)

    dirs1 = []
    dirs2 = []
    gains = []

    for v1, v2 in zip(B1, B2):
        # Magnitudes for gain
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue

        # One global phase for both sensors
        v1p, v2p = remove_global_phase_two_sensor(v1, v2)

        a1 = np.real(v1p)
        a2 = np.real(v2p)

        if np.linalg.norm(a1) == 0 or np.linalg.norm(a2) == 0:
            continue

        h1 = hat(a1)
        h2 = hat(a2)

        # Enforce a consistent global sign convention:
        # flip BOTH if needed, never individually.
        # Example: require h1's z >= 0 (any deterministic rule is fine).
        if h1[2] < 0:
            h1 = -h1
            h2 = -h2

        g = n2 / n1
        if not np.isfinite(g):
            continue

        dirs1.append(h1)
        dirs2.append(h2)
        gains.append(g)

    if not dirs1:
        raise ValueError("No valid SSP-derived features found.")

    dirs1 = np.asarray(dirs1)
    dirs2 = np.asarray(dirs2)
    gains = np.asarray(gains)

    # Features for clustering: [h1, h2, log(g)]
    feats = np.column_stack([dirs1, dirs2, np.log(gains)])
    X = StandardScaler().fit_transform(feats)

    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)

    # Keep largest n_keep non-noise clusters
    unique = [lab for lab in np.unique(labels) if lab >= 0]
    if not unique:
        raise ValueError("HDBSCAN found no clusters.")

    sizes = [(lab, np.sum(labels == lab)) for lab in unique]
    sizes.sort(key=lambda x: x[1], reverse=True)
    keep = [lab for lab, _ in sizes[:n_keep]]

    h1_dirs = []
    h2_dirs = []
    k_gain = []

    for lab in keep:
        idx = labels == lab
        if np.sum(idx) == 0:
            continue

        h1_mean = hat(dirs1[idx].mean(axis=0))
        h2_mean = hat(dirs2[idx].mean(axis=0))
        g_med = np.median(gains[idx])

        # Apply the same sign convention at cluster level
        if h1_mean[2] < 0:
            h1_mean = -h1_mean
            h2_mean = -h2_mean

        h1_dirs.append(h1_mean)
        h2_dirs.append(h2_mean)
        k_gain.append(g_med)

    return np.vstack(h1_dirs), np.vstack(h2_dirs), np.asarray(k_gain)

def dafgrad_clean(B, h1_dirs, h2_dirs, k_gain):
    """
    Dual-sensor DAFgrad in skewed frames.

    B        : (2, 3, N)
    h1_dirs  : (m, 3) unit disturbance directions at sensor 1
    h2_dirs  : (m, 3) unit disturbance directions at sensor 2
    k_gain   : (m,)  gains, |H2_k| / |H1_k| for each source k

    Returns:
        B_clean : (2, 3, N) cleaned signals (both sensors, identical in theory)
    """
    B = np.asarray(B, float)
    if B.shape[0] != 2 or B.shape[1] != 3:
        raise ValueError("B must have shape (2, 3, N)")

    h1_dirs = np.asarray(h1_dirs, float)
    h2_dirs = np.asarray(h2_dirs, float)
    k_gain = np.asarray(k_gain, float)

    if h1_dirs.shape != h2_dirs.shape:
        raise ValueError("h1_dirs and h2_dirs must have same shape")

    m = h1_dirs.shape[0]
    if k_gain.shape[0] != m:
        raise ValueError("k_gain length must match number of sources")

    # Unit direction vectors
    h1 = np.array([hat(v) for v in h1_dirs])
    h2 = np.array([hat(v) for v in h2_dirs])

    # Reconstruct disturbance vectors up to scale:
    # choose |H1_k| = 1, |H2_k| = k_gain[k]
    H1 = np.array([h1[k] for k in range(m)])
    H2 = np.array([k_gain[k] * h2[k] for k in range(m)])

    # Difference disturbance vectors and scaling factors
    HDelta = H1 - H2
    hD = np.array([hat(v) for v in HDelta])

    denom = np.linalg.norm(HDelta, axis=1)
    if np.any(denom < 1e-12):
        raise ValueError("Degenerate H1 - H2 for some source")

    alpha1 = np.linalg.norm(H1, axis=1) / denom
    alpha2 = np.linalg.norm(H2, axis=1) / denom

    # Build skewed frames: columns aligned with disturbance directions
    T1_inv = make_skew_frame(list(h1))
    T2_inv = make_skew_frame(list(h2))
    TD_inv = make_skew_frame(list(hD))

    T1 = np.linalg.inv(T1_inv)
    T2 = np.linalg.inv(T2_inv)
    TD = np.linalg.inv(TD_inv)

    # Project signals
    B1 = B[0].T        # (N,3)
    B2 = B[1].T
    BD = B1 - B2

    B1_p = (T1 @ B1.T).T
    B2_p = (T2 @ B2.T).T
    BD_p = (TD @ BD.T).T

    # Subtract along disturbance-aligned components
    B1_p_clean = B1_p.copy()
    B2_p_clean = B2_p.copy()
    for k in range(m):
        B1_p_clean[:, k] -= alpha1[k] * BD_p[:, k]
        B2_p_clean[:, k] -= alpha2[k] * BD_p[:, k]

    # Transform back
    B1_c = (T1_inv @ B1_p_clean.T).T
    B2_c = (T2_inv @ B2_p_clean.T).T

    B_clean = np.empty_like(B)
    B_clean[0] = B1_c.T
    B_clean[1] = B2_c.T
    return B_clean