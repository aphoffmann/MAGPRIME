"""
Author: Alex Hoffmann, 
Date: 05/22/2023
Description: Magnetic Noise removal simulation in a NEMISIS configuration.
"""
import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import random
from scipy.signal import chirp
import scipy.spatial.transform as st
import tqdm
from magprime.utility import interpolation
import os
from magprime import utility

"Noise Reduction Algorithms"
from magprime.algorithms import WAICUP

def noiseReactionWheel(fs, N, base_freq, seed):
    np.random.seed(seed) 
    shift_freq = np.random.uniform(.01, base_freq)
    duration = int(np.random.uniform(5, 200))
    time_of_shift = np.random.randint(0, (N/fs) - (2*duration))

    # Define time array.
    t = np.arange(N) / fs

    # Create signal.
    signal_rw = np.sin(2 * np.pi * base_freq * t)

    # Create down-chirp signal.
    down_chirp_signal = chirp(t[:duration*fs], base_freq, duration, shift_freq, method='hyperbolic')
    # Create up-chirp signal
    up_chirp_signal = chirp(t[:duration*fs], shift_freq, duration, base_freq, method='hyperbolic')

    # Modify original signal with chirp signal.
    signal_rw[time_of_shift*fs:(time_of_shift+duration)*fs] = down_chirp_signal
    signal_rw[(time_of_shift+duration)*fs:(time_of_shift+2*duration)*fs] = up_chirp_signal

    return signal_rw

def noiseMichibiki():
    "Import the magnetometer data from the file"
    B = utility.load_michibiki_data()
    interference = np.squeeze(B[1] - B[0])
    return(interference)

def noiseArcjet(N, seed = 0):
    # Defining a function that takes an integer N and an optional integer seed as parameters
    "Randomly Turn on and off signals" # A docstring that describes the function
    f = 500 # Setting a constant f
    np.random.seed(seed) # Setting the random seed for reproducibility
    zeros_ = np.random.choice([0, 1], size=(N // f + 1,), p=[3./10, 7./10]) # Generating an array of 0's and 1's with a 10% chance of 0 and a 90% chance of 1, with a size of N // f + 1
    ones_ = np.ones(N) # Generating an array of 1's with a size of N
    for i in range(N//f + 1): # Looping through the indices of the zeros_ array
        if(zeros_[i] == 1): # If the element at the current index is 1
            ones_[i*f:min((i+1)*f, N)] = 0 # Set the corresponding f elements or the remaining elements in the ones_ array to 0
    return(ones_) # Return the ones_ array as the output

def createMixingMatrix(seed, axis = 0):
    random.seed(seed)
    np.random.seed(seed)

    "Create Sensors"
    s1 = magpy.Sensor(position=(0,1348,1848), style_size=1.8)
    s2 = magpy.Sensor(position=(0,1207,1707), style_size=1.8)
    s3 = magpy.Sensor(position=(0,1065,1565), style_size=1.8)
    s = [s1,s2,s3]

    "Create Sources"
    "Create Sources"
    d1 = magpy.current.Loop(current=100, diameter=100, orientation=st.Rotation.random(),  position=(random.randint(-450, 450), random.randint(-450, 450), random.randint(10, 900)))
    d2 = magpy.current.Loop(current=100, diameter=100, orientation=st.Rotation.random(), position=(random.randint(-450, 450), random.randint(-450, 450), random.randint(10, 900)))    
    d3 = magpy.current.Loop(current=100, diameter=100, orientation=st.Rotation.random(), position=(random.randint(-450, 450), random.randint(-450, 450), random.randint(10, 900)))
    d4 = magpy.current.Loop(current=100, diameter=100, orientation=st.Rotation.random(), position=(random.randint(-450, 450), random.randint(-450, 450), random.randint(10, 900))) 
    src = [d1,d2,d3,d4]
    if False: plotNoiseFields(s,src)

    mixingMatrix = np.zeros((5,len(s)))
    mixingMatrix[0] = np.ones(len(s))

    for i in range(len(src)):
        mixing_vector = (src[i].getB(s)*1e6).T[axis]
        mixingMatrix[i+1] = mixing_vector

    return(mixingMatrix.T)

def plotNoiseFields(sensors, noises):
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))

    # Create grid
    ts1 = np.linspace(-500, 1500, 600)  # Extending the range in y
    ts2 = np.linspace(-500, 2000, 600)  # Extending the range in z
    grid = np.array([[(0, y, z) for y in ts1] for z in ts2])
    B = np.zeros((len(ts2), len(ts1), 3))

    # Calculate the magnetic field
    for n in noises:
        B += magpy.getB(n, grid) * 1e6

    Bamp = np.linalg.norm(B, axis=2)

    # Plot heatmap
    heatmap = ax.imshow(
        np.log(Bamp), 
        extent=(np.min(ts1), np.max(ts1), np.min(ts2), np.max(ts2)), 
        origin='lower', 
        aspect='auto', 
        cmap='Spectral_r'
    )


    # Plot noise sources and sensors
    for s in noises:
        ax.plot(s.position[1], s.position[2], 'bo', markersize=8)  # Plot using y and z
    for s in sensors:
        ax.plot(s.position[1], s.position[2], 'ro', markersize=8)  # Plot using y and z

    plt.xlabel("y [mm]")
    plt.ylabel("z [mm]")
    plt.colorbar(heatmap, ax=ax, label='Magnetic Field Magnitude [nT]')
    plt.show()

def randomizeSignals(N, seed = 0):
    # Defining a function that takes an integer N and an optional integer seed as parameters
    "Randomly Turn on and off signals" # A docstring that describes the function
    f = 300 # Setting a constant f to 300
    np.random.seed(seed) # Setting the random seed for reproducibility
    zeros_ = np.random.choice([0, 1], size=(N // f + 1,), p=[1./10, 9./10]) # Generating an array of 0's and 1's with a 10% chance of 0 and a 90% chance of 1, with a size of N // f + 1
    ones_ = np.ones(N) # Generating an array of 1's with a size of N
    for i in range(N//f + 1): # Looping through the indices of the zeros_ array
        if(zeros_[i] == 1): # If the element at the current index is 1
            ones_[i*f:min((i+1)*f, N)] = 0 # Set the corresponding f elements or the remaining elements in the ones_ array to 0
    return(ones_) # Return the ones_ array as the output

def snr(x, y):
  # x and y are numpy arrays of the same length
  # x is the original signal and y is the noisy signal
  # returns the SNR in decibels
  x_mean = np.mean(x) # calculate the mean of x
  num = np.sum((x - x_mean)**2) # calculate the numerator of SNR
  den = np.sum((x - y)**2) # calculate the denominator of SNR
  if den == 0: # avoid division by zero
    return np.inf # return infinity if denominator is zero
  else:
    return 10 * np.log10(num / den) # return SNR in decibels

def create_gaps(N, seed):
    np.random.seed(seed)
    buffer = int(0.1 * N)  # Buffer to exclude outer 10% from being gapped

    # Generate an array of 1's with a size of N
    gaps = np.ones(N)
    start_idx = buffer
    end_idx = N - buffer
    valid_positions = list(range(start_idx, end_idx))
    num_gaps = np.random.randint(3,15)

    if num_gaps > 0:
        gap_indices = np.random.choice(valid_positions, size=num_gaps, replace=False)
        gap_indices.sort()  # Sort the indices to manage them in order

        # Introduce gaps of random sizes into the data array
        previous_end = 0
        for idx in gap_indices:
            if idx < previous_end:
                continue  # Skip this gap to avoid overlap
            gap_size = np.random.randint(1, 50)  # Random gap size between 1 and 100
            end_gap = min(idx + gap_size, N)
            gaps[idx:end_gap] = 0
            previous_end = end_gap  # Update the end of the last gap

    return(gaps)

def run():
    "Create Signals Arrays"
    duration = 5*3600; sampleRate = 1; N = duration*sampleRate; T = 1/sampleRate
    samples = np.linspace(0, duration, N, endpoint=False)
    signals = np.zeros((5, samples.shape[0]))

    "Import ambient magnetic field signal."
    swarm = utility.load_swarm_data(160000, 160000 + N)

    "Import Michibiki Data"
    michibiki = noiseMichibiki()

    if os.path.exists("interp_results.csv"):
        # Read the existing CSV file and get the last seed value
        results = pd.read_csv("interp_results.csv")
        last_seed = results["seed"].iloc[-1]
    else:
        # Create an empty data frame with columns
        results = pd.DataFrame(columns=['seed', 'rmse_waicup','rmse_interp', "rmse_b1", "rmse_b2", "rmse_b3",
                                         'corr_waicup','corr_interp', "corr_b1", "corr_b2", "corr_b3",
                                        'snr_waicup','snr_interp', "snr_b1", "snr_b2", "snr_b3"])
        last_seed = -1 # Set the last seed to -1 if the CSV file does not exist

    "BEGIN MONTE CARLO SIMULATION"
    for i in tqdm.tqdm(range(last_seed + 1, 1)):
        random.seed(i)
        np.random.seed(i)

        n = random.randint(0, michibiki.shape[-1]-N)

        "Create Source Signals"
        signals[0] = swarm[2]
        signals[1] = noiseReactionWheel(sampleRate, N, np.random.uniform(.1,sampleRate/2), i) # Reaction Wheels
        signals[2] = (michibiki[2][n:n+N]-np.mean(michibiki[2][n:n+N]))/np.max(np.abs((michibiki[2][n:n+N]-np.mean(michibiki[2][n:n+N]))))
        signals[3] = noiseArcjet(N, i) # Arcjet
        signals[4] = signal.sawtooth(2 * np.pi * .1 * samples)*randomizeSignals(N,i)

        "Create Mixing Matrices"
        Kz = createMixingMatrix(i, 2)

        "Create Mixed Signals"
        B = Kz @ signals

        "WAICUP"
        WAICUP.fs = sampleRate
        WAICUP.uf = 1500
        WAICUP.detrend = True
        B_waicup = WAICUP.clean(np.copy(B), triaxial = False)

        rmse_waicup = np.sqrt(((swarm[2]-B_waicup)**2).mean(axis=0))
        corr_waicup = np.corrcoef(swarm[2], B_waicup)[0,1]
        snr_waicup = snr(swarm[2], B_waicup)

        "Run MSSA interpolation Scheme"
        gaps = create_gaps(N, i)
        B_gap = np.copy(B)
        B_gap[:, gaps == 0] = np.nan
        B_interpolated = interpolation.mssa.interpolate(B_gap, gaps, triaxial=False)
        B_waicup_interp = WAICUP.clean(np.copy(B_interpolated), triaxial = False)

        rmse_interp = np.sqrt(((swarm[2]-B_waicup_interp)**2).mean(axis=0))
        corr_interp = np.corrcoef(swarm[2], B_waicup_interp)[0,1]
        snr_interp = snr(swarm[2], B_waicup_interp)

        "Run linear interpolation Scheme"
        gaps = create_gaps(N, i)
        B_gap = np.copy(B)
        B_gap[:, gaps == 0] = np.nan
        B_lin_interpolated = interpolation.linear.interpolate(B_gap[:,0,:], gaps, triaxial=False)
        B_lin_waicup_interp = WAICUP.clean(np.copy(B_lin_interpolated), triaxial=False)
        
        rmse_lin_interp = np.sqrt(((swarm[2]-B_lin_waicup_interp)**2).mean(axis=0))
        corr_lin_interp = np.corrcoef(swarm[2], B_lin_waicup_interp)[0,1]
        snr_lin_interp = snr(swarm[2], B_lin_waicup_interp)

        "Run zero fill interpolation Scheme"
        gaps = create_gaps(N, i)
        B_gap = np.copy(B)
        B_gap[:, gaps == 0] = 0
        B_zero_waicup = WAICUP.clean(np.copy(B_gap), triaxial = False)

        rmse_zero_interp = np.sqrt(((swarm[2]-B_zero_waicup)**2).mean(axis=0))
        corr_zero_interp = np.corrcoef(swarm[2], B_zero_waicup)[0,1]
        snr_zero_interp = snr(swarm[2], B_zero_waicup)

        "No Noise Removal"
        rmse_b1 = np.sqrt(((swarm[2]-B[0])**2).mean(axis=0))
        rmse_b2 = np.sqrt(((swarm[2]-B[1])**2).mean(axis=0))
        rmse_b3 = np.sqrt(((swarm[2]-B[2])**2).mean(axis=0))
        corr_b1 = np.corrcoef(swarm[2], B[0])[0,1]
        corr_b2 = np.corrcoef(swarm[2], B[1])[0,1]
        corr_b3 = np.corrcoef(swarm[2], B[2])[0,1]
        snr_b1 = snr(swarm[2], B[0])
        snr_b2 = snr(swarm[2], B[1])
        snr_b3 = snr(swarm[2], B[2])

        "Save all results to a csv file"
        results = results.append({
                                    "seed": i,
                                    "rmse_waicup": rmse_waicup,
                                    "rmse_interp": rmse_interp,
                                    "rmse_lin_interp": rmse_lin_interp,
                                    "rmse_zero_interp": rmse_zero_interp,
                                    "rmse_b1": rmse_b1,
                                    "rmse_b2": rmse_b2,
                                    "rmse_b3": rmse_b3,
                                    "corr_waicup": corr_waicup,
                                    "corr_interp": corr_interp,
                                    "corr_lin_interp": corr_lin_interp,
                                    "corr_zero_interp": corr_zero_interp,
                                    "corr_b1": corr_b1,
                                    "corr_b2": corr_b2,
                                    "corr_b3": corr_b3,
                                    "snr_waicup": snr_waicup,
                                    "snr_interp": snr_interp,
                                    "snr_lin_interp": snr_lin_interp,
                                    "snr_zero_interp": snr_zero_interp,
                                    "snr_b1": snr_b1,
                                    "snr_b2": snr_b2,
                                    "snr_b3": snr_b3
                                }, ignore_index=True)
        results.to_csv("interp_results.csv", index=False) 



if __name__ == "__main__":
    run()