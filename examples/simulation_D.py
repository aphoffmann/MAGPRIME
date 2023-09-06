"""
Author: Alex Hoffmann, 
Date: 05/22/2023
Description: Magnetic Noise removal simulation in a boomless three magnetometer configuration.
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
from magprime import ICA
from magprime import WAICUP
from magprime import MSSA

def createMixingMatrix(axis = 0):
    "Create Sensors"
    s1 = magpy.Sensor(position=(40,40,0), style_size=1.8)
    s2 = magpy.Sensor(position=(-40,40,0), style_size=1.8)
    s3 = magpy.Sensor(position=(40,-40,0), style_size=1.8)
    s4 = magpy.Sensor(position=(-40,-40,0), style_size=1.8)
    
    s = [s1,s2,s3,s4]
    
    "Create Sources"
    d1 = magpy.current.Loop(current=-.51, diameter=10,  position=(43, 16, 27))
    d3 = magpy.current.Loop(current=.35, diameter=10,  position=(-36, 26, 83))
    d4 = magpy.current.Loop(current=.20, diameter=10,  position=(25, -32, 66))    
    d5 = magpy.current.Loop(current=-.75, diameter=10,  position=(-25, -13, 37)) 
    
    mixingMatrix = np.zeros((5,4))
    mixingMatrix[0] = np.ones(4)
    src = [d1,d3,d4,d5]
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
    Bx[0] += np.random.normal(0, 5, N); Bx[1] += np.random.normal(0, 5, N); Bx[2] += np.random.normal(0, 5, N); Bx[3] += np.random.normal(0, 5, N)
    By[0] += np.random.normal(0, 5, N); By[1] += np.random.normal(0, 5, N); By[2] += np.random.normal(0, 5, N); By[3] += np.random.normal(0, 5, N)
    Bz[0] += np.random.normal(0, 5, N); Bz[1] += np.random.normal(0, 5, N); Bz[2] += np.random.normal(0, 5, N); Bz[3] += np.random.normal(0, 5, N)
    
    "Create B"
    B = np.array([Bx,By,Bz])
    B = np.swapaxes(B,0,1)

    "WAICUP"
    WAICUP.fs = sampleRate
    WAICUP.uf = 400
    WAICUP.dj = 0.25
    WAICUP.denoise = False
    #B_waicup = WAICUP.clean(np.copy(B))

    "MSSA"
    MSSA.window_size = 500
    MSSA.uf = 500
    MSSA.detrend = True
    #B_mssa = MSSA.clean(np.copy(B))


    "ICA"
    ICA.fs = 50
    ICA.uf = 400
    ICA.detrend = True
    B_ica = ICA.clean(np.copy(B))
    plt.plot(B_ica.T)
    plt.show()

    "UBSS"







    "REAM"

    "Plot Results"




if __name__ == "__main__":
    run()