"""
Author: Alex Hoffmann, 
Date: 04/20/2023
Description: Testing Ream Gradiometry
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
from magprime import REAM, PiCoG

def run():
    df=pd.read_csv('examples\SPACE_DATA\mstac2e2esupp1.csv', sep=',',header=0)
    samples = np.arange(0, len(df['Bin_x']))/16
    B = df[['Bin_x', 'Bin_y', 'Bin_z', 'Bout_x', 'Bout_y', 'Bout_z']].to_numpy().T
    B = B.reshape(2, 3, -1)
    sampleRate = 16

    REAM.p = 98
    REAM.delta_B = 0.1
    REAM.n = 10
    #%% Plot Mixed Signals
    fig, ax = plt.subplots(3,1)
    ax[0].set_title("Mixed Signals", fontsize = '16')
    for i in range(3):
        ax[i].set_ylabel('nT', fontsize = '12')
        ax[i].plot(samples, B[1,i,:])
        ax[i].tick_params(labelsize='8' )
        if(i != 2): plt.setp(ax[i].get_xticklabels(), visible=False)
    ax[2].set_xlabel('seconds', fontsize = '12')
    plt.show()

    amb_mf = REAM.clean(B) # shape (3, n_samples)

    #%% Plot Cleaned Signals
    fig, ax = plt.subplots(3,1)
    ax[0].set_title("Cleaned Signals", fontsize = '16')
    for i in range(3):
        ax[i].set_ylabel('nT', fontsize = '12')
        ax[i].plot(samples, amb_mf[i,:])
        ax[i].tick_params(labelsize='8' )
        if(i != 2): plt.setp(ax[i].get_xticklabels(), visible=False)
    ax[2].set_xlabel('seconds', fontsize = '12')
    plt.show()

    
    # PiCoG
    B_picog = PiCoG.clean(B)

    #%% Plot Cleaned Signals
    fig, ax = plt.subplots(3,1)
    ax[0].set_title("PiCoG Cleaned Signals", fontsize = '16')
    for i in range(3):
        ax[i].set_ylabel('nT', fontsize = '12')
        ax[i].plot(samples, B_picog[i,:])
        ax[i].tick_params(labelsize='8' )
        if(i != 2): plt.setp(ax[i].get_xticklabels(), visible=False)
    ax[2].set_xlabel('seconds', fontsize = '12')
    plt.show()

    return

if __name__ == "__main__":
    run()