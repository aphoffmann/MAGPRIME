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
import os
from scipy.signal import butter, lfilter
from magprime import utility
from invertiblewavelets import DyadicFilterBank, Cauchy

"Noise Reduction Algorithms"
from magprime.algorithms import ICA, MSSA, NESS, PiCoG, SHEINKER, REAM, UBSS, WAICUP

"Parameters"
alpha_couplings = None # Coupling matrix between the sensors and sources for NESS


def noiseReactionWheel(fs, N, base_freq, seed):
    np.random.seed(seed) 
    shift_freq = np.random.uniform(1, base_freq)
    duration = int(np.random.uniform(1, 5))
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
    s1 = magpy.Sensor(position=(.050,.050,.300), style_size=1.8)
    s2 = magpy.Sensor(position=(-.050,0,.200), style_size=1.8)
    s3 = magpy.Sensor(position=(-.050,-.050,0), style_size=1.8)
    s = [s1,s2,s3]

    "Create Sources"
    d1 = magpy.current.Loop(current=10, diameter=.010, orientation=st.Rotation.random(),  position=(random.uniform(-.040, .040), random.uniform(-.040, .040), random.uniform(.010, .0290)))
    d2 = magpy.current.Loop(current=10, diameter=.010, orientation=st.Rotation.random(), position=(random.uniform(-.040, .040), random.uniform(-.040, .040), random.uniform(.010, .0290)))    
    d3 = magpy.current.Loop(current=10, diameter=.010, orientation=st.Rotation.random(), position=(random.uniform(-.040, .040), random.uniform(-.040, .040), random.uniform(.010, .0290)))
    d4 = magpy.current.Loop(current=10, diameter=.010, orientation=st.Rotation.random(), position=(random.uniform(-.040, .040), random.uniform(-.040, .040), random.uniform(.010, .0290))) 
    src = [d1,d2,d3,d4]

    "Calculate Couplings"
    global alpha_couplings;
    alpha_couplings = np.sum(s1.getB(src), axis = 0)/ np.sum(s2.getB(src), axis = 0)

    mixingMatrix = np.zeros((5,len(s)))
    mixingMatrix[0] = np.ones(len(s))

    for i in range(len(src)):
        mixing_vector = (src[i].getB(s)*1e9).T[axis]
        mixingMatrix[i+1] = mixing_vector

    return(mixingMatrix.T)

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

def run():
    "Create Signals Arrays"
    duration = 100; sampleRate = 50; N = duration*sampleRate; T = 1/sampleRate
    samples = np.linspace(0, duration, N, endpoint=False)
    signals = np.zeros((5, samples.shape[0]))

    "Import ambient magnetic field signal."
    swarm = utility.load_swarm_data(160000, 160000 + N)

    "Import Michibiki Data"
    michibiki = noiseMichibiki()

    if os.path.exists("magprime_results_B.csv"):
        # Read the existing CSV file and get the last seed value
        results = pd.read_csv("magprime_results_B.csv")
        last_seed = results["seed"].iloc[-1]
    else:
        # Create an empty data frame with columns
        results = pd.DataFrame(columns=['seed', 'rmse_ica', 'rmse_mssa', 'rmse_ness', 'rmse_picog', 'rmse_sheinker', 'rmse_ream', 'rmse_ubss', 'rmse_waicup', "rmse_b1", "rmse_b2", "rmse_b3",
                                        'corr_ica', 'corr_mssa', 'corr_ness', 'corr_picog', 'corr_sheinker', 'corr_ream', 'corr_ubss', 'corr_waicup', "corr_b1", "corr_b2", "corr_b3",
                                        'snr_ica', 'snr_mssa', 'snr_ness', 'snr_picog', 'snr_sheinker', 'snr_ream', 'snr_ubss', 'snr_waicup', "snr_b1", "snr_b2", "snr_b3"])
        last_seed = -1 # Set the last seed to -1 if the CSV file does not exist

    "BEGIN MONTE CARLO SIMULATION"
    axes = ['x', 'y', 'z']
    rows = []
    for i in tqdm.tqdm(range(last_seed + 1, 100)):
        random.seed(i)
        np.random.seed(i)

        n = random.randint(0, michibiki.shape[-1]-5000)

        "Create Source Signals"
        signals[0] = swarm[0]
        signals[1] = noiseReactionWheel(sampleRate, N, np.random.randint(4,sampleRate//2), i) # Reaction Wheels
        signals[2] = (michibiki[0][n:n+5000]-np.mean(michibiki[0][n:n+5000]))/np.max(np.abs((michibiki[0][n:n+5000]-np.mean(michibiki[0][n:n+5000])))) # Michibiki
        signals[3] = noiseArcjet(N, i) # Arcjet
        signals[4] = signal.sawtooth(2 * np.pi * 3 * samples)*randomizeSignals(N,random.randint(0,100000))

        "Create Mixing Matrices"
        Kx = createMixingMatrix(i, 0)
        Ky = createMixingMatrix(i, 1)
        Kz = createMixingMatrix(i, 2)

        "Create Mixed Signals"
        Bx = Kx @ signals

        signals[0] = swarm[1]
        signals[2] = (michibiki[1][n:n+5000]-np.mean(michibiki[1][n:n+5000]))/np.max(np.abs((michibiki[1][n:n+5000]-np.mean(michibiki[1][n:n+5000]))))
        By = Ky @ signals

        signals[0] = swarm[2]
        signals[2] = (michibiki[2][n:n+5000]-np.mean(michibiki[2][n:n+5000]))/np.max(np.abs((michibiki[2][n:n+5000]-np.mean(michibiki[2][n:n+5000]))))
        Bz = Kz @ signals

        "Create B"
        B = np.array([Bx,By,Bz])
        B = np.swapaxes(B,0,1)

        "SHEINKER"
        # No Detrend
        B_sheinker = SHEINKER.clean(np.copy(B))

        rmse_sheinker = np.sqrt(((swarm.T-B_sheinker.T)**2).mean(axis=0))
        corr_sheinker = np.zeros(3); snr_sheinker = np.zeros(3)
        for j in range(3):
            corr_sheinker[j] = np.corrcoef(swarm[j], B_sheinker[j])[0,1]
            snr_sheinker[j] = snr(swarm[j], B_sheinker[j])

        "NESS"
        # No Detrend
        NESS.aii = alpha_couplings
        B_ness = NESS.clean(np.copy(B))

        rmse_ness = np.sqrt(((swarm.T-B_ness.T)**2).mean(axis=0))
        corr_ness = np.zeros(3); snr_ness = np.zeros(3)
        for j in range(3):
            corr_ness[j] = np.corrcoef(swarm[j], B_ness[j])[0,1]
            snr_ness[j] = snr(swarm[j], B_ness[j])

        "WAICUP"
        fb = DyadicFilterBank(
            wavelet=Cauchy(100), 
            fs=50, N=10000, 
            real=True, 
            dj = 1/12, 
            s_max = 20,
            compensation=False)

        WAICUP.filterbank = fb
        WAICUP.detrend = False
        B_waicup = WAICUP.clean(np.copy(B))

        rmse_waicup = np.sqrt(((swarm.T-B_waicup.T)**2).mean(axis=0))
        corr_waicup = np.zeros(3); snr_waicup = np.zeros(3)
        for j in range(3):
            corr_waicup[j] = np.corrcoef(swarm[j], B_waicup[j])[0,1]
            snr_waicup[j] = snr(swarm[j], B_waicup[j])


        "PiCoG"
        PiCoG.order = 3
        B_picog = PiCoG.clean(np.copy(B))
        
        rmse_picog = np.sqrt(((swarm.T-B_picog.T)**2).mean(axis=0))
        corr_picog = np.zeros(3); snr_picog = np.zeros(3)
        for j in range(3):
            corr_picog[j] = np.corrcoef(swarm[j], B_picog[j])[0,1]
            snr_picog[j] = snr(swarm[j], B_picog[j])

        "ICA"
        ICA.uf = 500
        ICA.detrend = True
        B_ica = ICA.clean(np.copy(B))

        rmse_ica = np.sqrt(((swarm.T-B_ica.T)**2).mean(axis=0))
        corr_ica = np.zeros(3); snr_ica = np.zeros(3)
        for j in range(3):
            corr_ica[j] = np.corrcoef(swarm[j], B_ica[j])[0,1]
            snr_ica[j] = snr(swarm[j], B_ica[j])

        "REAM"
        REAM.n = 50
        REAM.delta_B = 5
        B_ream = REAM.clean(np.copy(B))

        rmse_ream = np.sqrt(((swarm.T-B_ream.T)**2).mean(axis=0))
        corr_ream = np.zeros(3); snr_ream = np.zeros(3)
        for j in range(3):
            corr_ream[j] = np.corrcoef(swarm[j], B_ream[j])[0,1]      
            snr_ream[j] = snr(swarm[j], B_ream[j])  

        "UBSS"
        UBSS.sigma = 15
        UBSS.lambda_ = 2
        UBSS.fs = sampleRate
        UBSS.bpo = 2
        UBSS.cs_iters = 2
        B_ubss = UBSS.clean(np.copy(B))

        rmse_ubss = np.sqrt(((swarm.T-B_ubss.T)**2).mean(axis=0))
        corr_ubss = np.zeros(3); snr_ubss = np.zeros(3)
        for j in range(3):
            corr_ubss[j] = np.corrcoef(swarm[j], B_ubss[j])[0,1]
            snr_ubss[j] = snr(swarm[j], B_ubss[j])

        "MSSA"
        MSSA.window_size = 100
        MSSA.uf = 500
        MSSA.detrend = True
        B_mssa = MSSA.clean(np.copy(B))
        
        rmse_mssa = np.sqrt(((swarm.T-B_mssa.T)**2).mean(axis=0))
        corr_mssa = np.zeros(3); snr_mssa = np.zeros(3)
        for j in range(3):
            corr_mssa[j] = np.corrcoef(swarm[j], B_mssa[j])[0,1]    
            snr_mssa[j] = snr(swarm[j], B_mssa[j])        

        "No Noise Removal"
        rmse_b1 = np.sqrt(((swarm.T-B[0].T)**2).mean(axis=0))
        rmse_b2 = np.sqrt(((swarm.T-B[1].T)**2).mean(axis=0))
        rmse_b3 = np.sqrt(((swarm.T-B[2].T)**2).mean(axis=0))
        corr_b1 = np.zeros(3); snr_b1 = np.zeros(3)
        corr_b2 = np.zeros(3); snr_b2 = np.zeros(3)
        corr_b3 = np.zeros(3); snr_b3 = np.zeros(3)
        for j in range(3):
            corr_b1[j] = np.corrcoef(swarm[j], B[0][j])[0,1]
            corr_b2[j] = np.corrcoef(swarm[j], B[1][j])[0,1]
            corr_b3[j] = np.corrcoef(swarm[j], B[2][j])[0,1]
            snr_b1[j] = snr(swarm[j], B[0][j])
            snr_b2[j] = snr(swarm[j], B[1][j])
            snr_b3[j] = snr(swarm[j], B[2][j])

        "Save all results to a csv file"
        for k, ax in enumerate(axes):
            rows.append({
                'seed': i,
                'axis': ax,
                'rmse_ica': rmse_ica[k],
                'rmse_mssa': rmse_mssa[k],
                'rmse_ness': rmse_ness[k],
                'rmse_picog': rmse_picog[k],
                'rmse_sheinker': rmse_sheinker[k],
                'rmse_ream': rmse_ream[k],
                'rmse_ubss': rmse_ubss[k],
                'rmse_waicup': rmse_waicup[k],
                'rmse_b1': rmse_b1[k],
                'rmse_b2': rmse_b2[k],
                "rmse_b3": rmse_b3[k],
                'corr_ica': corr_ica[k],
                'corr_mssa': corr_mssa[k],
                'corr_ness': corr_ness[k],
                'corr_picog': corr_picog[k],
                'corr_sheinker': corr_sheinker[k],
                'corr_ream': corr_ream[k],
                'corr_ubss': corr_ubss[k],
                'corr_waicup': corr_waicup[k],
                'corr_b1': corr_b1[k],
                'corr_b2': corr_b2[k],
                'corr_b3': corr_b3[k],
                'snr_ica': snr_ica[k],
                'snr_mssa': snr_mssa[k],
                'snr_ness': snr_ness[k],
                'snr_picog': snr_picog[k],
                'snr_sheinker': snr_sheinker[k],
                'snr_ream': snr_ream[k],
                'snr_ubss': snr_ubss[k],
                'snr_waicup': snr_waicup[k],
                'snr_b1': snr_b1[k],
                'snr_b2': snr_b2[k],
                'snr_b3': snr_b3[k]})
    
            results = pd.DataFrame(rows)
            results.to_csv("magprime_results_B.csv", index=False) 


if __name__ == "__main__":
    run()