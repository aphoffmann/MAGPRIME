# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:33:33 2022

@author: Alex
"""

"""
Author: Alex Hoffmann, 
Date: 11/2/2022
Description: Replication of FastICA experiment by Imajo et al. (2021) on a 
             single axis.
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
import ICA
import MSSA
import SHEINKER
from WAIC import WaveletAdaptiveInterferenceCancellation as WAIC
import UBSS as U
import scaleogram as scg
import collections

def createMixingMatrix(axis = 0, show = False, plot = False):
    "Create Sensors"
    s1 = magpy.Sensor(position=(300,250,100), style_size=1.8)
    s2 = magpy.Sensor(position=(250,250,400), style_size=1.8)
    s3 = magpy.Sensor(position=(200,250,100), style_size=1.8)
    s = [s1,s2,s3]
    
    "Create Sources"
    d1 = magpy.current.Loop(current=.46, diameter=500,  position=(50,250,10))
    d2 = magpy.current.Loop(current=-.31, diameter=400,  position=(150,250,300))    
    d3 = magpy.current.Loop(current=.38, diameter=200,  position=(450,250,200)) 

    if(show): magpy.show([s1,s2,s3,d1,d2,d3])
    if(plot): plotNoiseFields([s1,s2, s3],[d1,d2,d3])
    
    mixingMatrix = np.zeros((4,3))
    mixingMatrix[0] = np.ones(3)
    src = [d1,d2,d3]
    for i in range(len(src)):
        mixing_vector = (src[i].getB(s)*1e6).T[axis]
        mixingMatrix[i+1] = mixing_vector

    return(mixingMatrix.T)

def plotNoiseFields(sensors, noises):
    fig, ax = plt.subplots(1, 1, figsize=(13,5))

    # create grid
    ts1 = np.linspace(0, 500, 250); ts2 = np.linspace(0, 1200, 600);
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
    "Randomly Turn on and off signals"
    f = 100
    np.random.seed(seed)
    zeros_ = np.random.choice([0, 1], size=(N // f,), p=[1./8, 7./8])
    ones_ = np.ones(N)
    for i in range(N//f):
        if(zeros_[i] == 1):
            ones_[i*f:(i+1)*f] = 0
    return(ones_)

def run():
    #%% Create Signals Arrays
    duration = 100; sampleRate = 50; N = duration*sampleRate; T = 1/sampleRate
    samples = np.linspace(0, duration, N, endpoint=False)
    signals = np.zeros((4, samples.shape[0]))
    
    #%% Import 50 Hz SWARM Residual Data
    "Import 50 Hz SWARM Residual Data"
    df=pd.read_csv('SPACE_DATA\Swarm_MAGA_HR_20150317_0900.csv', sep=',',header=None)
    r = df[10]
    swarm3 = np.array([np.fromstring(r[i][1:-1], dtype=float, sep=' ') for i in range(1, r.shape[0])])
    swarm3 = swarm3.T
    swarm3 = swarm3[:,160000:160000+N]
    swarm = swarm3[0]
    
    #%% Create Source Signals
    signals[0] = swarm
    signals[1] = np.sin(2 * np.pi * 2 * samples)*np.sin(2 * np.pi * 20 * samples) # Reaction Wheels
    signals[2] = np.sin(2 * np.pi * 6 * samples)*randomizeSignals(N,1)*np.random.normal(0, 1, N) #signal.square(2 * np.pi * 2 * samples)*randomizeSignals(N,1)
    signals[3] = signal.square(2 * np.pi * 4 * samples)*randomizeSignals(N,4)
    
    """
    #%% Plot Source Signals
    fig, ax = plt.subplots(4,1)
    ax[0].set_title("Source Signals", Fontsize = '16')
    for i in range(4):
        ax[i].set_ylabel('nT', Fontsize = '12')
        ax[i].plot(samples, signals[i])
        ax[i].tick_params(labelsize='8' )
        if(i != 3): plt.setp(ax[i].get_xticklabels(), visible=False)
    ax[3].set_xlabel('seconds', Fontsize = '12')
    """
    
    #%% Mix Signals
    mixingMatrix = createMixingMatrix(0, False, False)
    B = mixingMatrix @ signals 
    B[0] += np.random.normal(0, 3.9, N)
    B[1] += np.random.normal(0, .1, N)
    B[2] += np.random.normal(0, 3.9, N)
    print("BOOM: ", np.round(mean_squared_error(B[1], swarm, squared=False),3), " nT")

    """
    #%% Plot Mixed Signals
    fig, axs = plt.subplots(B.shape[0],1)
    axs[0].set_title("Mixed Signals", Fontsize = '16')
    for i in range(B.shape[0]):
        axs[i].plot(samples, B[i])
        axs[i].set_ylabel('nT', Fontsize = '12')
        axs[i].tick_params(labelsize='8' )
        if(i != 1):  plt.setp(axs[i].get_xticklabels(), visible=False)
    axs[1].set_xlabel('seconds', Fontsize = '12')
    
    #%% Boom
    print("BOOM: ", np.round(mean_squared_error(B[1], swarm, squared=False),3), " nT")
    
    "Noise Signal"
    fig, ax = plt.subplots(1,1, figsize=(14, 6))
    data = B[1] - np.mean(B[1])
    time = np.linspace(0,100, swarm.shape[-1])
    wavelet='cmor0.7-1.5' 
    ax = scg.cws(time, data, scales=np.geomspace(1, 5000,200), wavelet=wavelet,
            ax=ax, cmap="Spectral_r", cbar=None, ylabel="Frequency [Hz]",
            xlabel=" ", title="Noisy Signal", cbarlabel = "nT$^2$/Hz",
            yscale='log', yaxis='frequency', coi=True, spectrum='power', cscale='log', clim=(100, 1.5e6))
    mappable = ax.collections[0]
    fig.colorbar(mappable=mappable, ax=ax)
    """
    if(False):
        #%% ICA
        "Create ICA Signals"
        mixedSignalsX = np.copy(B)
        mixingMatrixY = createMixingMatrix(1, False, False)
        signals[0] = swarm3[1]
        mixedSignalsY = mixingMatrixY @ signals
        mixingMatrixZ = createMixingMatrix(2, False, False)
        signals[0] = swarm3[2]
        mixedSignalsZ = mixingMatrixZ @ signals
        mixedSignals = np.vstack((mixedSignalsX,mixedSignalsY, mixedSignalsZ))
        signal_ica = ICA.clean(np.copy(mixedSignals), method = "TriAxis")[0]
        print("ICA: ", np.round(mean_squared_error(signal_ica.real, swarm, squared=False),3), " nT")
        
        #%% SHEINKER
        signal_sheinker = SHEINKER.clean(np.copy(B)[1:])
        print("SHEINKER: ", np.round(mean_squared_error(signal_sheinker.real, swarm, squared=False),3), " nT")

        #%% WAIC
        W = WAIC(fs = 50, dj = 0.05)
        signal_waic = W.clean(np.copy(B), detrend = True, method = "iterativeWAIC", lp = 4*100)
        print("WAIC: ", np.round(mean_squared_error(signal_waic.real[200:-200], swarm[200:-200], squared=False),3), " nT")

        #%% MSSA
        signal_mssa = MSSA.clean(np.copy(B), detrend = True, window_size = 400, uf = 400, alpha = 0.01)
        print("MSSA: ", np.round(mean_squared_error(signal_mssa.real[200:-200], swarm[200:-200], squared=False),3), " nT")        

    if(True):
        #%% UBSS
        U.setMagnetometers(3)
        U.Q = 10
        U.lowf = 0.03
        U.sigma = 3.9
        U.lambda_ = 2
        U.sspTol = 2
        U.sampleRate = 50
        #U.updateDBSCAN(eps=.1, minSamples=3)
        U.weight = 4.5
        
        U.clusterNSGT(np.copy(B))
        
        signal_ubss = U.demixNSGT(np.copy(B), weighted=True)[0]
        print("UBSS: ", np.round(mean_squared_error(signal_ubss.real, swarm, squared=False),3), " nT")
        return
                

        

        #%% UBSS
        U.clusterCentroids = collections.OrderedDict((i, val) for i,val in enumerate((mixingMatrix/np.linalg.norm(mixingMatrix, axis = 0)).T))
        U.clusterCentroids[0] = np.ones(3)
        signal_ubss = U.demixNSGT(np.copy(B), weighted=True)[0]
        print("UBSS: ", np.round(mean_squared_error(signal_ubss.real, swarm, squared=False),3), " nT")
        
        fig, ax = plt.subplots(1,1, figsize=(14, 6))
        data = signal_ubss - np.mean(signal_ubss)
        time = np.linspace(0,100, swarm.shape[-1])
        wavelet='cmor0.7-1.5' 
        ax = scg.cws(time, data, scales=np.geomspace(1, 5000,200), wavelet=wavelet,
                ax=ax, cmap="Spectral_r", cbar=None, ylabel="Frequency [Hz]",
                xlabel=" ", title="Noisy Signal", cbarlabel = "nT$^2$/Hz",
                yscale='log', yaxis='frequency', coi=True, spectrum='power', cscale='log', clim=(100, 1.5e6))
        mappable = ax.collections[0]
        fig.colorbar(mappable=mappable, ax=ax)

        fig = plt.subplots(1,1)
        plt.plot(swarm)
        plt.plot(signal_ubss)
        
    #fig = plt.subplots(1,1)
    #plt.plot(swarm)
    #plt.plot(signal_waic)
    

if __name__ == '__main__':
    run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    