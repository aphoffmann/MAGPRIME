import numpy as np
from scipy.ndimage import uniform_filter1d
import itertools
from nsgt import CQ_NSGT

"General Parameters"
uf = 400            # Uniform Filter Size for detrending
detrend = True     # Detrend the data

"Algorithm Parameters"
fs = 1              # Sampling Frequency
bins = 4            # NSGT scale spacing (we may adjust this for NSGT)
lowest_freq = None  # Lowest frequency for NSGT
boom = None         # Trend to use during retrending process

def clean(B, triaxial=True):
    """
    B: magnetic field measurements from the sensor array (n_sensors, axes, n_samples)
    triaxial: boolean for whether to use triaxial or uniaxial ICA
    """

    if triaxial:
        result = np.zeros((3, B.shape[-1]))
        for axis in range(3):
            result[axis] = cleanWAICUP(B[:, axis, :])
        return result
    else:
        "B: (n_sensors, n_samples)"
        result = cleanWAICUP(B)
        return result

def cleanWAICUP(sensors):
    "Detrend"
    if detrend:
        trend = uniform_filter1d(sensors, size=uf)
        sensors = sensors - trend

    # Initialize NSGT
    nsgt = CQ_NSGT(fmin=lowest_freq, fmax=fs/2, bins=bins, fs=fs, Ls=sensors.shape[-1], real=False, matrixform=True)
    

    result = dual(sensors, nsgt)


    "Retrend"
    if detrend:
        if boom is not None:
            result += trend[boom]
        else:
            result += np.mean(trend, axis=0)

    return result

def dual(sig, nsgt):
    "Transform signals into NSGT domain"

    c_sig0 = np.array(nsgt.forward(sig[0]))  # NSGT coefficients for sensor 1
    c_sig1 = np.array(nsgt.forward(sig[1]))  # NSGT coefficients for sensor 2
    d = c_sig1 - c_sig0
    # --- Compute Cross-Correlations for K_hat ---
    numerator = np.sum(d * c_sig1, axis=1)       # Cross-correlation
    denominator = np.sum(d*c_sig0, axis=1)           # Auto-correlation

    # --- Estimate Scaling Factor K_hat ---
    denominator = np.where(denominator == 0, 1e-10, denominator)  # Prevent division by zero
    k_hat = numerator / denominator                            # Shape: (num_frequencies,)
    k_hat  = k_hat ** 100

    # --- Compute Clean NSGT Coefficients ---
    k_hat_reshaped = k_hat[:, np.newaxis]                     # Shape: (num_frequencies, 1)
    clean_numerator = k_hat_reshaped * c_sig0 - c_sig1        # Numerator for A_hat
    clean_denominator = k_hat_reshaped - 1                    # Denominator for A_hat

    # Compute the clean NSGT coefficients
    clean_c = clean_numerator / clean_denominator             # Shape: (num_frequencies, num_time_frames)

    # --- Inverse NSGT to Obtain Clean Signal ---
    clean_sig = nsgt.backward(clean_c)                         # Cleaned time-domain signal
    clean_sig = np.real(clean_sig)

    return clean_sig

def multi(sig, nsgt):
    "Find Combinations"
    pairs = list(itertools.combinations([i for i in range(sig.shape[0])], 2))
    waicup_level1 = np.zeros((len(pairs), sig.shape[-1]))

    for i in range(len(pairs)):
        waicup_level1[i] = dual(np.vstack((sig[pairs[i][0]], sig[pairs[i][1]])), nsgt)

    "Reconstruct Ambient Magnetic Field Signal"
    W_n = waicup_level1
    Y_00 = np.mean([nsgt.backward(W_n[i]) for i in range(len(pairs))], axis=0)
    
    "Return Ambient Magnetic Field"
    return Y_00
