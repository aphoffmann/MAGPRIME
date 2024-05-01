import numpy as np
from pymssa import MSSA 

"MSSA Parameters"
n_components = "variance_threshold"
variance_explained_threshold = 0.95
pa_percentile_threshold = None
svd_method = 'randomized'
verbose = False

def interpolate(B,gaps, triaxial = False):
    """
    Perform interpolation through Singular Spectrum Analysis (SSA)
    Input:
        B: magnetic field measurements from the sensor array (n_sensors, n_samples)
        gaps: binary mask of gaps in the data (n_sensors, n_samples) e.g. [1,1,1,1,0,0,0,0,1,1,1,1] - 0 is missing data
        triaxial: boolean flag to indicate if the data is triaxial or monoaxial
    Output:
        result: reconstructed ambient field without gaps (n_sensors, n_samples)
    """
    if(triaxial):
        result = np.zeros(B.shape)
        for i in range(3):
            result[:,i,:] = monoaxial_interpolation(B[:,i,:], gaps)
    else:
        result = monoaxial_interpolation(B, gaps)

    return result

def monoaxial_interpolation(B, gaps):
    """
    Perform interpolation through Singular Spectrum Analysis (SSA)
    Input:
        B: magnetic field measurements from the sensor array (n_sensors, n_samples)
        gaps: binary mask of gaps in the data (n_sensors, n_samples) e.g. [1,1,1,1,0,0,0,0,1,1,1,1] - 0 is missing data
    Output:
        result: reconstructed ambient field without gaps (n_sensors, n_samples)
    """
    n_samples = B.shape[1]
    result = np.copy(B)

    # Find Gap indices
    gap_indices = np.where(gaps == 0)[0]
    if gap_indices.size == 0:
        return result  # No gaps to fill

    gap_starts = np.hstack(([0], np.where(np.diff(gap_indices) != 1)[0] + 1))
    gap_ends = np.hstack((np.where(np.diff(gap_indices) != 1)[0], [len(gap_indices) - 1]))
    full_gaps = [(gap_indices[start], gap_indices[end]) for start, end in zip(gap_starts, gap_ends)]
    

    # Calculate window size L based on the size of the data and the gap
    for (gap_start, gap_end) in full_gaps:
        gap_length = gap_end - gap_start + 1
        L = 2*(gap_end-gap_start)

        # Get segments surrounding the gap for forward and backward prediction
        pre_gap_data = B[:, :gap_start]
        post_gap_data = B[:, gap_end + 1:]

        # Perform forward M-SSA forecast on the data
        L = int(min(L, pre_gap_data.shape[1]//2))
        mssa = MSSA(n_components = n_components,
                    variance_explained_threshold = variance_explained_threshold,
                    pa_percentile_threshold = pa_percentile_threshold,
                    svd_method = 'randomized',
                    window_size=L,
                    verbose = verbose)
        mssa.fit(pre_gap_data.T)
        fwd_fc = mssa.forecast(timepoints_out = gap_length)

        # Perform backward M-SSA forecast on the data
        L = int(min(L, post_gap_data.shape[1]//2))
        post_gap_data = np.flip(post_gap_data, axis=1)
        mssa = MSSA(n_components = n_components,
                    variance_explained_threshold = variance_explained_threshold,
                    pa_percentile_threshold = pa_percentile_threshold,
                    svd_method = 'randomized',
                    window_size=L,
                    verbose = verbose)
        mssa.fit(post_gap_data.T)
        bwd_fc = mssa.forecast(timepoints_out = gap_length)

        # Fill the gap with exponential weights
        filled_gap = fill_gap_with_exponential_weights(fwd_fc, np.flip(bwd_fc, axis=1))
        result[:, gap_start:gap_end + 1] = filled_gap

    return result

def fill_gap_with_exponential_weights(fwd_fc, bwd_fc):
    """
    Calculate the weighted average of the forward and backward forecast to fill the gap
    Input:
        fwd_fc: forward forecast of the gap (n_sensors, gap_length)
        bwd_fc: backward forecast of the gap (n_sensors, gap_length)
    Output:
        X_filled: filled gap (n_sensors, gap_length)
    """
    n = fwd_fc.shape[1]
    i = np.arange(n) / n
    w_fwd = (1-i)**2
    w_bwd = (1-np.flip(i))**2

    w_sum = w_fwd + w_bwd
    w_fwd_normalized = w_fwd / w_sum
    w_bwd_normalized = w_bwd / w_sum

    X_filled = w_fwd_normalized * fwd_fc + w_bwd_normalized * bwd_fc
    return X_filled
