import numpy as np
from pymssa import MSSA 
from tqdm import tqdm

"MSSA Parameters"
n_components = "svht"
variance_explained_threshold = 0.95
pa_percentile_threshold = None
svd_method = 'randomized'
verbose = True

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

    # Find Gap starting and ending indices and window validity
    gap_info = find_gaps(gaps)
    

    # Calculate window size L based on the size of the data and the gap
    for gap_start, gap_end, gap_length, _ in tqdm(gap_info):
        L = 2*gap_length
        gaps[gap_start:gap_end+1] = 1
        gap_info = find_gaps(gaps)

        # Get segments surrounding the gap for forward and backward prediction
        window_start = max(0, gap_start - L)
        window_end = min(n_samples - 1, gap_end + L)
        pre_gap_data = result[:, window_start:gap_start]
        post_gap_data = result[:, gap_end + 1:window_end + 1]

        # check if window touches the edge of the data
        if(gap_end == n_samples - 1 or gap_start == 0):
            result[:, gap_start:gap_end + 1] = 0
            continue

        # Check if gaps in pre_gap_data or post_gap_data
        window_invalid = np.sum(1-gaps[window_start:gap_start]) + np.sum(1- gaps[gap_end + 1:window_end + 1])


        # If gap_length is less than 10, use linear interpolation
        if gap_length < 10 or window_invalid > 0:
            # Linear interpolation
            for sensor in range(result.shape[0]):
                result[sensor, gap_start:gap_end+1] = np.interp(
                                np.arange(gap_start, gap_end +1),
                                np.array([gap_start-1,gap_end+1]),
                                result[sensor, np.array([gap_start-1,gap_end+1])]
                            )
            continue

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


def find_gaps(gaps):
    # Find Gap starting and ending indices
    gap_indices = np.where(gaps == 0)[0]
    if gap_indices.size == 0:
        return []  # No gaps to fill

    gap_starts = np.hstack(([0], np.where(np.diff(gap_indices) != 1)[0] + 1))
    gap_ends = np.hstack((np.where(np.diff(gap_indices) != 1)[0], [len(gap_indices) - 1]))
    full_gaps = [(gap_indices[start], gap_indices[end]) for start, end in zip(gap_starts, gap_ends)]

    # Create a list to store gap info with window analysis
    gap_info = []

    for start, end in full_gaps:
        gap_length = end - start + 1
        window_radius = 2 * gap_length
        window_start = max(0, start - window_radius)
        window_end = min(len(gaps) - 1, end + window_radius)

        # Count valid data points in the window
        window_data_count = np.sum(gaps[window_start:start]) + np.sum(gaps[end + 1:window_end + 1])

        gap_info.append((start, end, gap_length, int(window_data_count//gap_length > 3)))

    # Rank gaps - example by smallest gap first, then by maximum surrounding data
    gap_info.sort(key=lambda x: (-x[3],x[2]))
    return(gap_info)