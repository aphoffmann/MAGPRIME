import numpy as np
from tqdm import tqdm

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
    for gap_start, gap_end, _, _ in tqdm(gap_info):
        gaps[gap_start:gap_end+1] = 1

        # check if window touches the edge of the data
        if(gap_end == n_samples - 1 or gap_start == 0):
            result[:, gap_start:gap_end + 1] = 0
            continue

        # Linear interpolation
        for sensor in range(result.shape[0]):
            result[sensor, gap_start:gap_end+1] = np.interp(
                            np.arange(gap_start, gap_end +1),
                            np.array([gap_start-1,gap_end+1]),
                            result[sensor, np.array([gap_start-1,gap_end+1])]
                        )
    return result





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