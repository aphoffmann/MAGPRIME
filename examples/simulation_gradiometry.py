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
from magprime import *

def createMixingMatrix(axis = 0):
    "Create Sensors"
    s1 = magpy.Sensor(position=(250,250,350), style_size=1.8)
    s2 = magpy.Sensor(position=(250,250,700), style_size=1.8)
    s = [s1,s2]

    "Create Sources"
    d1 = magpy.current.Loop(current=.61, diameter=500,  position=(50,250,10))
    d2 = magpy.current.Loop(current=-.41, diameter=400,  position=(150,250,300))    
    d3 = magpy.current.Loop(current=.51, diameter=200,  position=(450,250,200)) 

    mixingMatrix = np.zeros((4,2,3))
    mixingMatrix[0] = np.ones((2,3))
    src = [d1,d2,d3]
    for i in range(len(src)):
        mixing_vector = (src[i].getB(s)*1e6)
        mixingMatrix[i+1] = mixing_vector
        mixingMatrix[:,i+1,:] = mixing_vector.T

    return(mixingMatrix.T)

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
    signals[1] = np.sin(2 * np.pi * 2 * samples)*np.sin(2 * np.pi * 20 * samples) # Reaction Wheels
    signals[2] = np.sin(2 * np.pi * 6 * samples)*randomizeSignals(N,1)*np.random.normal(0, 1, N)
    signals[3] = signal.square(2 * np.pi * 4 * samples)*randomizeSignals(N,4)
    
    """
    #%% Plot Source Signals
    fig, ax = plt.subplots(4,1)
    ax[0].set_title("Source Signals", fontsize = '16')
    for i in range(4):
        ax[i].set_ylabel('nT', fontsize = '12')
        ax[i].plot(samples, signals[i])
        ax[i].tick_params(labelsize='8' )
        if(i != 3): plt.setp(ax[i].get_xticklabels(), visible=False)
    ax[3].set_xlabel('seconds', fontsize = '12')
    """
    
    #%% Mix Signals
    mixingMatrix = createMixingMatrix()
    B = mixingMatrix @ signals 
    B = np.transpose(B, axes=(1,0,2))
    B[0] += np.random.normal(0, 3.9, N)
    B[1] += np.random.normal(0, .1, N)
    print("BOOM: ", np.round(mean_squared_error(B[1][0], swarm, squared=False),3), " nT")

    

if __name__ == '__main__':
    run()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    