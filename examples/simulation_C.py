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
from sklearn.metrics import mean_squared_error

"Noise Reduction Algorithms"
from magprime import ICA, MSSA, NESS, PiCoG, SHEINKER, REAM, UBSS, WAICUP

"Parameters"
alpha_couplings = None # Coupling matrix between the sensors and sources for NESS

def createMixingMatrix(axis = 0):
    "Create Sensors"
    s1 = magpy.Sensor(position=(300,250,200), style_size=1.8)
    s2 = magpy.Sensor(position=(200,250,200), style_size=1.8)
    s3 = magpy.Sensor(position=(200,250,10), style_size=1.8)
    s = [s1,s2,s3]
    
    "Create Sources"
    d1 = magpy.current.Loop(current=6.1, diameter=50,  position=(200,250,110))
    d2 = magpy.current.Loop(current=-4.1, diameter=40,  position=(250,250,160))    
    d3 = magpy.current.Loop(current=5.1, diameter=20,  position=(270,250,80))
    d4 = magpy.current.Loop(current=-5.1, diameter=30,  position=(300,250,50)) 
    src = [d1,d2,d3,d4]
    plotNoiseFields(s,src)

    "Calculate Couplings"
    global alpha_couplings;
    alpha_couplings = np.sum(s1.getB(src), axis = 0)/ np.sum(s2.getB(src), axis = 0)

    mixingMatrix = np.zeros((5,len(s)))
    mixingMatrix[0] = np.ones(len(s))

    for i in range(len(src)):
        mixing_vector = (src[i].getB(s)*1e6).T[axis]
        mixingMatrix[i+1] = mixing_vector*10

    return(mixingMatrix.T)


def plotNoiseFields(sensors, noises):
    fig, ax = plt.subplots(1, 1, figsize=(13,5))

    # create grid
    ts1 = np.linspace(100, 400, 450); ts2 = np.linspace(0, 400, 400);
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
    Bx[0] += np.random.normal(0, 5, N); Bx[1] += np.random.normal(0, 5, N); Bx[2] += np.random.normal(0, 5, N); 
    By[0] += np.random.normal(0, 5, N); By[1] += np.random.normal(0, 5, N); By[2] += np.random.normal(0, 5, N); 
    Bz[0] += np.random.normal(0, 5, N); Bz[1] += np.random.normal(0, 5, N); Bz[2] += np.random.normal(0, 5, N); 
    
    
    "Create B"
    B = np.array([Bx,By,Bz])
    B = np.swapaxes(B,0,1)

    "SHEINKER"
    SHEINKER.uf = 500
    SHEINKER.detrend = True
    B_sheinker = SHEINKER.clean(np.copy(B))

    "NESS"
    NESS.uf = 500
    NESS.detrend = True
    NESS.aii = alpha_couplings
    B_ness = NESS.clean(np.copy(B))

    "PiCoG"
    PiCoG.uf = 500
    PiCoG.detrend = True
    B_picog = PiCoG.clean(np.copy(B))

    "REAM"
    REAM.uf = 500
    REAM.detrend = True
    REAM.delta_B = 10
    B_ream = REAM.clean(np.copy(B))

    "UBSS"
    UBSS.boom = 1
    UBSS.sigma = 25
    UBSS.lambda_ = 2
    UBSS.fs = sampleRate
    UBSS.bpo = 5
    B_ubss = UBSS.clean(np.copy(B))

    "WAICUP"
    WAICUP.fs = sampleRate
    WAICUP.uf = 500
    WAICUP.detrend = True
    B_waicup = WAICUP.clean(np.copy(B))

    "ICA"
    ICA.uf = 500
    ICA.detrend = True
    B_ica = ICA.clean(np.copy(B))

    "MSSA"
    MSSA.window_size = 500
    MSSA.uf = 500
    MSSA.detrend = True
    B_mssa = MSSA.clean(np.copy(B))   

    "Calculate RMSE of inner [500:-500] for each algorithm for each axis with respect to the ambient magnetic field, swarm, and also do the boom B[1]"
    rmse = np.zeros((3,9))
    rmse[0,0] = mean_squared_error(swarm[0,500:-500], B_sheinker[0,500:-500], squared=False)
    rmse[0,1] = mean_squared_error(swarm[0,500:-500], B_ness[0,500:-500], squared=False)
    rmse[0,2] = mean_squared_error(swarm[0,500:-500], B_picog[0,500:-500], squared=False)
    rmse[0,3] = mean_squared_error(swarm[0,500:-500], B_ream[0,500:-500], squared=False)
    rmse[0,4] = mean_squared_error(swarm[0,500:-500], B_ubss[0,500:-500], squared=False)
    rmse[0,5] = mean_squared_error(swarm[0,500:-500], B_waicup[0,500:-500], squared=False)
    rmse[0,6] = mean_squared_error(swarm[0,500:-500], B_ica[0,500:-500], squared=False)
    rmse[0,7] = mean_squared_error(swarm[0,500:-500], B_mssa[0,500:-500], squared=False)
    rmse[0,8] = mean_squared_error(swarm[0,500:-500], B[1,0,500:-500], squared=False)

    rmse[1,0] = mean_squared_error(swarm[1,500:-500], B_sheinker[1,500:-500], squared=False)
    rmse[1,1] = mean_squared_error(swarm[1,500:-500], B_ness[1,500:-500], squared=False)
    rmse[1,2] = mean_squared_error(swarm[1,500:-500], B_picog[1,500:-500], squared=False)
    rmse[1,3] = mean_squared_error(swarm[1,500:-500], B_ream[1,500:-500], squared=False)
    rmse[1,4] = mean_squared_error(swarm[1,500:-500], B_ubss[1,500:-500], squared=False)
    rmse[1,5] = mean_squared_error(swarm[1,500:-500], B_waicup[1,500:-500], squared=False)
    rmse[1,6] = mean_squared_error(swarm[1,500:-500], B_ica[1,500:-500], squared=False)
    rmse[1,7] = mean_squared_error(swarm[1,500:-500], B_mssa[1,500:-500], squared=False)
    rmse[1,8] = mean_squared_error(swarm[1,500:-500], B[1,1,500:-500], squared=False)

    rmse[2,0] = mean_squared_error(swarm[2,500:-500], B_sheinker[2,500:-500], squared=False)
    rmse[2,1] = mean_squared_error(swarm[2,500:-500], B_ness[2,500:-500], squared=False)
    rmse[2,2] = mean_squared_error(swarm[2,500:-500], B_picog[2,500:-500], squared=False)
    rmse[2,3] = mean_squared_error(swarm[2,500:-500], B_ream[2,500:-500], squared=False)
    rmse[2,4] = mean_squared_error(swarm[2,500:-500], B_ubss[2,500:-500], squared=False)
    rmse[2,5] = mean_squared_error(swarm[2,500:-500], B_waicup[2,500:-500], squared=False)
    rmse[2,6] = mean_squared_error(swarm[2,500:-500], B_ica[2,500:-500], squared=False)
    rmse[2,7] = mean_squared_error(swarm[2,500:-500], B_mssa[2,500:-500], squared=False)
    rmse[2,8] = mean_squared_error(swarm[2,500:-500], B[1,2,500:-500], squared=False)

    "Print RMSE Results in a table"
    print(pd.DataFrame(rmse, columns = ['SHEINKER', 'NESS', 'PiCoG', 'REAM', 'UBSS', 'WAICUP', 'ICA', 'MSSA', 'BOOM'], index = ['X', 'Y', 'Z']))

    """
        SHEINKER        NESS       PiCoG       REAM       UBSS     WAICUP         ICA       MSSA  
    X   6.117220  13.8961299  341.708833  16.372984  5.8774314  10.954444   24.565284  24.887799  
    Y   4.366110  4.83097452  388.961276   4.953542   3.485227   7.112823   48.078402   2.505182 
    Z  14.887682  59.9939193  256.460033  25.710732  32.710411   9.381458   14.027635  19.596584  
    """

    "Create a stackplot of all of the algorithms results with swarm plotted over them. It should be 3x3"
    fig, axs = plt.subplots(3, 3, sharex=True, figsize=(10,10))
    axs[0, 0].plot(B_sheinker.T, label='SHEINKER')
    axs[0, 0].plot(swarm.T, color='black', label='Swarm')
    axs[0, 0].set_title('SHEINKER')
    axs[0, 1].plot(B_ness.T, label='NESS')
    axs[0, 1].plot(swarm.T, color='black', label='Swarm')
    axs[0, 1].set_title('NESS')

    axs[1, 0].plot(B_ream.T, label='REAM')
    axs[1, 0].plot(swarm.T, color='black', label='Swarm')
    axs[1, 0].set_title('REAM')
    axs[1, 1].plot(B_ubss.T, label='UBSS')
    axs[1, 1].plot(swarm.T, color='black', label='Swarm')
    axs[1, 1].set_title('UBSS')
    axs[1, 2].plot(B_waicup.T, label='WAICUP')
    axs[1, 2].plot(swarm.T, color='black', label='Swarm')
    axs[1, 2].set_title('WAICUP')
    axs[2, 0].plot(B_ica.T, label='ICA')
    axs[2, 0].plot(swarm.T, color='black', label='Swarm')
    axs[2, 0].set_title('ICA')
    axs[2, 1].plot(B_mssa.T, label='MSSA')
    axs[2, 1].plot(swarm.T, color='black', label='Swarm')
    axs[2, 1].set_title('MSSA')
    axs[2, 2].plot(B[1].T, label='Boom')
    axs[2, 2].plot(swarm.T, color='black', label='Swarm')
    axs[2, 2].set_title('Boom')
    

    plt.show()

    # Save Plot
    fig.savefig('examples\simulation_B.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    run()