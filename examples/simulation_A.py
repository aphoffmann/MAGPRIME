"""
Author: Alex Hoffmann, 
Date: 05/22/2023
Description: Magnetic Noise removal simulation in a gradiometry configuration.
"""
import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

"Noise Reduction Algorithms"
from magprime import ICA, MSSA, NESS, PiCoG, SHEINKER, REAM, UBSS, WAICUP

def createMixingMatrix(axis = 0):
    "Create Sensors"
    s1 = magpy.Sensor(position=(250,250,350), style_size=1.8)
    s2 = magpy.Sensor(position=(250,250,700), style_size=1.8)
    s = [s1,s2]
    
    "Create Sources"
    d1 = magpy.current.Loop(current=6.1, diameter=50,  position=(50,250,10))
    d2 = magpy.current.Loop(current=-4.1, diameter=40,  position=(150,250,60))    
    d3 = magpy.current.Loop(current=5.1, diameter=20,  position=(450,250,20))
    d4 = magpy.current.Loop(current=-5.1, diameter=30,  position=(300,250,50)) 

    
    mixingMatrix = np.zeros((5,len(s)))
    mixingMatrix[0] = np.ones(len(s))
    src = [d1,d2,d3,d4]
    for i in range(len(src)):
        mixing_vector = (src[i].getB(s)*1e6).T[axis]
        mixingMatrix[i+1] = mixing_vector*10

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

def run():
    "Create Signals Arrays"
    duration = 100; sampleRate = 50; N = duration*sampleRate; T = 1/sampleRate
    samples = np.linspace(0, duration, N, endpoint=False)
    signals = np.zeros((5, samples.shape[0]))

    "Import ambient magnetic field signal."
    df=pd.read_csv('examples\SPACE_DATA\Swarm_MAGA_HR_20150317_0900.csv', sep=',',header=None)
    r = df[10]
    swarm3 = np.array([np.fromstring(r[i][1:-1], dtype=float, sep=' ') for i in range(1, r.shape[0])])
    swarm3 = swarm3.T
    swarm = swarm3[:,160000:160000+N]

    "Create Source Signals"
    signals[0] = swarm[0]
    signals[1] = np.sin(2 * np.pi * 15 * samples) + np.sin(2 * np.pi * 20 * samples) # Reaction Wheels
    signals[2] = signal.sawtooth(2 * np.pi * .3 * samples)*randomizeSignals(N,5)
    signals[3] = signal.square(2 * np.pi * 5 * samples)*randomizeSignals(N,7)
    signals[4] = np.sin(2 * np.pi * 3 * samples)*randomizeSignals(N,10)

    "Create Mixing Matrices"
    Kx = createMixingMatrix(0)
    Ky = createMixingMatrix(1)
    Kz = createMixingMatrix(2)

    "Create Mixed Signals"
    Bx = Kx @ signals
    
    signals[0] = swarm[1]
    By = Ky @ signals

    signals[0] = swarm[2]
    Bz = Kz @ signals

    "Add Noise"
    Bx[0] += np.random.normal(0, 5, N); Bx[1] += np.random.normal(0, 5, N); 
    By[0] += np.random.normal(0, 5, N); By[1] += np.random.normal(0, 5, N); 
    Bz[0] += np.random.normal(0, 5, N); Bz[1] += np.random.normal(0, 5, N); 
    
    "Create B"
    B = np.array([Bx,By,Bz])
    B = np.swapaxes(B,0,1)

    "SHEINKER"
    B_sheinker = SHEINKER.clean(np.copy(B))

    "NESS"
    NESS.r1 = 350
    NESS.r2 = 700
    B_ness = NESS.clean(np.copy(B))

    "PiCoG"
    B_picog = PiCoG.clean(np.copy(B))

    "REAM"
    REAM.delta_B = 10
    B_ream = REAM.clean(np.copy(B))

    "UBSS"
    "WAICUP"
    "ICA"
    "MSSA"
    
    

if __name__ == "__main__":
    run()