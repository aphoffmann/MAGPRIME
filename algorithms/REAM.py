
import numpy as np
from scipy.signal import windows, welch
from scipy.signal import find_peaks

def gradiometry_filter(B1, B2, fs, delta_B, n, p):
    """
    Perform magnetic gradiometry using frequency-domain filtering
    Input:
        B1: magnetic field measurements from the inboard sensor (Nx3 array)
        B2: magnetic field measurements from the outboard sensor (Nx3 array)
        fs: sampling frequency (Hz)
        delta_B: threshold for the change in the differenced field envelope (nT)
        n: number of time steps for the change in the envelope
        p: percentile threshold for identifying spectral peaks (0-100)
    Output:
        B_amb: reconstructed ambient field without the spacecraft-generated fields (Nx3 array)
    """
    # Calculate the differenced field
    B_diff = B2 - B1
    
    # Calculate the rolling maximum and minimum of the differenced field
    delta_B_series = pd.Series(B_diff)
    B_max = delta_B_series.rolling(1920, center=True, min_periods=1).max()
    B_min = delta_B_series.rolling(1920, center=True, min_periods=1).min()
    
    # Identify the intervals where the differenced field changes significantly
    dB = np.abs(B_max - B_min) / n # change in the envelope
    mask = (dB > delta_B).to_numpy() # boolean mask for the threshold condition
    intervals = [] # list of intervals
    start = None # start index of an interval
    for i in range(len(mask)):
        if mask[i] and start is None: # start of an interval
            start = i
        elif (not mask[i] or i == len(mask)-1) and start is not None: # end of an interval
            end = i
            if end - start >= n: # check if the interval is long enough
                intervals.append((start, end))
            start = None # reset start index
    
    # Initialize the output array
    B_amb = np.copy(B1)
    
        # Loop over each interval
    for start, end in intervals:
        win = windows.kaiser(end - start, beta=10)
        B1_win = B1[start:end] * win # windowed inboard sensor data
        B2_win = B2[start:end] * win # windowed outboard sensor data
        B_diff_win = B_diff[start:end] * win # windowed differenced data
        
        # Perform the FFT
        F1 = np.fft.fft(B1_win)
        F2 = np.fft.fft(B2_win)
        F_diff = np.fft.fft(B_diff_win)

        # Compute the power spectra
        P1 = np.abs(F1)**2
        P2 = np.abs(F2)**2
        P_diff = np.abs(F_diff)**2

        # Identify the spectral peaks in the differenced field spectrum using a percentile threshold
        thresh = np.percentile(P_diff, 99)
        peak_indices, _ = find_peaks(P_diff, height=thresh)

        # Plot index of the peaks
        plt.plot(P_diff); plt.plot(peak_indices, P_diff[peak_indices], marker="x")
        plt.show()

        # Suppress the peaks in the inboard and outboard sensor spectra
        F1_suppr = F1.copy()
        F2_suppr = F2.copy()

        for peak in peak_indices:
            F1_suppr[peak] *= 0.01
            F2_suppr[peak] *= 0.01

        # Perform the inverse FFT to obtain the suppressed time-domain signals
        B1_suppr = np.fft.ifft(F1_suppr) # / win
        B2_suppr = np.fft.ifft(F2_suppr) # / win
        # Reconstruct the ambient field at each sensor
        B_amb[start:end] = (B1_suppr + B2_suppr) / 2

        # Plot B_amb
        fig, ax = plt.subplots(1,1)
        plt.plot(B_amb)
        plt.show()