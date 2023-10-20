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

"Noise Reduction Algorithms"
from magprime import ICA, MSSA, NESS, PiCoG, SHEINKER, REAM, UBSS, WAICUP

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
    qzs_1 = np.loadtxt(r"michibiki.dat", dtype=np.float, usecols=(0,4,5,6,7,8,9))
    B_qzs = qzs_1.T

    "Subtract the bias from the magnetometer data"
    B_qzs[1] -= 60 # MAM-S1 X-Axis
    B_qzs[2] -= 410 # MAM-S1 Y-Axis
    B_qzs[3] -= -202 # MAM-S1 Z-Axis
    B_qzs[4] -= -528 # MAM-S2 X-Axis
    B_qzs[5] -= -200 # MAM-S2 Y-Axis
    B_qzs[6] -= 478 # MAM-S2 Z-Axis

    Bx = B_qzs[4] - B_qzs[1]
    By = B_qzs[5] - B_qzs[2]
    Bz = B_qzs[6] - B_qzs[3]
    michibiki = np.vstack((Bx,By,Bz))
    #michibiki -= uniform_filter1d(michibiki, size=500, axis = -1)
    return(michibiki)

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
    s1 = magpy.Sensor(position=(50,50,300), style_size=1.8)
    s2 = magpy.Sensor(position=(-50,0,200), style_size=1.8)
    s3 = magpy.Sensor(position=(-50,-50,0), style_size=1.8)
    s = [s1,s2,s3]

    "Create Sources"
    d1 = magpy.current.Loop(current=10, diameter=100, orientation=st.Rotation.random(),  position=(random.randint(-40, 40), random.randint(-40, 40), random.randint(10, 290)))
    d2 = magpy.current.Loop(current=10, diameter=100, orientation=st.Rotation.random(), position=(random.randint(-40, 40), random.randint(-40, 40), random.randint(10, 290)))    
    d3 = magpy.current.Loop(current=10, diameter=100, orientation=st.Rotation.random(), position=(random.randint(-40, 40), random.randint(-40, 40), random.randint(10, 290)))
    d4 = magpy.current.Loop(current=10, diameter=100, orientation=st.Rotation.random(), position=(random.randint(-40, 40), random.randint(-40, 40), random.randint(10, 290))) 
    src = [d1,d2,d3,d4]
    if False: plotNoiseFields(s,src)

    "Calculate Couplings"
    global alpha_couplings;
    alpha_couplings = np.sum(s1.getB(src), axis = 0)/ np.sum(s2.getB(src), axis = 0)

    mixingMatrix = np.zeros((5,len(s)))
    mixingMatrix[0] = np.ones(len(s))

    for i in range(len(src)):
        mixing_vector = (src[i].getB(s)*1e6).T[axis]
        mixingMatrix[i+1] = mixing_vector

    return(mixingMatrix.T)

def plotNoiseFields(sensors, noises):
    fig, ax = plt.subplots(1, 1, figsize=(13,5))

    # create grid
    ts1 = np.linspace(-200, 200, 450); ts2 = np.linspace(-100, 300, 600);
    grid = np.array([[(x,250,z) for x in ts1] for z in ts2])
    B = np.zeros(grid.shape)
    for n in noises:
        B += magpy.getB(n, grid)*1e6
       
        
    Bamp = np.linalg.norm(B, axis=2)
    #Bamp /= np.amax(Bamp)
    
    sp = ax.streamplot(
        grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
        density=3,
        color=np.log(Bamp),
        linewidth=1,
        cmap='coolwarm',
    )
    
    "Plot NoiseMakers"
    for s in noises:
        plt.plot(s.position[0], s.position[-1], 'bo')
    #plt.legend()
    
    "Plot Sensors"
    for s in sensors:
        plt.plot(s.position[0], s.position[-1], 'ro')
    #plt.legend()

    plt.xlabel("[mm]");    plt.ylabel("[mm]")

    plt.colorbar(sp.lines, ax=ax, label='[Log nT]')

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
    df=pd.read_csv('Swarm_MAGA_HR_20150317_0900.csv', sep=',',header=None)
    r = df[10]
    swarm = np.array([np.fromstring(r[i][1:-1], dtype=float, sep=' ') for i in range(1, r.shape[0])]).T[:,160000:160000+N]

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
    for i in tqdm.tqdm(range(last_seed + 1, 10)):
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
        cutoff_freq = 0.01; order = 4; nyquist = 0.5 * 50; normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff, btype='low')
        signals[1:] -= lfilter(b, a, signals[1:])
        Bx = Kx @ signals

        signals[0] = swarm[1]
        signals[2] = (michibiki[1][n:n+5000]-np.mean(michibiki[1][n:n+5000]))/np.max(np.abs((michibiki[1][n:n+5000]-np.mean(michibiki[1][n:n+5000]))))
        signals[1:] -= lfilter(b, a, signals[1:])
        By = Ky @ signals

        signals[0] = swarm[2]
        signals[2] = (michibiki[2][n:n+5000]-np.mean(michibiki[2][n:n+5000]))/np.max(np.abs((michibiki[2][n:n+5000]-np.mean(michibiki[2][n:n+5000]))))
        signals[1:] -= lfilter(b, a, signals[1:])
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
        WAICUP.fs = sampleRate
        WAICUP.uf = 500
        WAICUP.detrend = True
        B_waicup = WAICUP.clean(np.copy(B))

        rmse_waicup = np.sqrt(((swarm.T-B_waicup.T)**2).mean(axis=0))
        corr_waicup = np.zeros(3); snr_waicup = np.zeros(3)
        for j in range(3):
            corr_waicup[j] = np.corrcoef(swarm[j], B_waicup[j])[0,1]
            snr_waicup[j] = snr(swarm[j], B_waicup[j])


        "PiCoG"
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
        UBSS.boom = 1
        UBSS.sigma = 2
        UBSS.lambda_ = 2
        UBSS.fs = sampleRate
        UBSS.bpo = 10
        try:
            B_ubss = UBSS.clean(np.copy(B))
        except:
            UBSS.cs_iters = 1
            B_ubss = UBSS.clean(np.copy(B))
            
        rmse_ubss = np.sqrt(((swarm.T-B_ubss.T)**2).mean(axis=0))
        corr_ubss = np.zeros(3); snr_ubss = np.zeros(3)
        for j in range(3):
            corr_ubss[j] = np.corrcoef(swarm[j], B_ubss[j])[0,1]
            snr_ubss[j] = snr(swarm[j], B_ubss[j])

        "MSSA"
        MSSA.window_size = 500
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
        results = results.append({  "seed": i,
                                    "rmse_ica": rmse_ica,
                                    "rmse_mssa": rmse_mssa,
                                    "rmse_ness": rmse_ness,
                                    "rmse_picog": rmse_picog,
                                    "rmse_sheinker": rmse_sheinker,
                                    "rmse_ream": rmse_ream,
                                    "rmse_ubss": rmse_ubss,
                                    "rmse_waicup": rmse_waicup,
                                    "rmse_b1": rmse_b1,
                                    "rmse_b2": rmse_b2,
                                    "rmse_b3": rmse_b3,
                                    "corr_ica": corr_ica,
                                    "corr_mssa": corr_mssa,
                                    "corr_ness": corr_ness,
                                    "corr_picog": corr_picog,
                                    "corr_sheinker": corr_sheinker,
                                    "corr_ream": corr_ream,
                                    "corr_ubss": corr_ubss,
                                    "corr_waicup": corr_waicup,
                                    "corr_b1": corr_b1,
                                    "corr_b2": corr_b2,
                                    "corr_b3": corr_b3,
                                    "snr_ica": snr_ica,
                                    "snr_mssa": snr_mssa,
                                    "snr_ness": snr_ness,
                                    "snr_picog": snr_picog,
                                    "snr_sheinker": snr_sheinker,
                                    "snr_ream": snr_ream,
                                    "snr_ubss": snr_ubss,
                                    "snr_waicup": snr_waicup,
                                    "snr_b1": snr_b1,
                                    "snr_b2": snr_b2,
                                    "snr_b3": snr_b3}, ignore_index=True)
        results.to_csv("magprime_results_B.csv", index=False) 


if __name__ == "__main__":
    run()