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
from algorithms import REAM
from algorithms import MVA

def run():
    df=pd.read_csv('SPACE_DATA\mstac2e2esupp1.csv', sep=',',header=0)
    samples = np.arange(0, len(df['Bin_x']))/16
    B = np.vstack((df['Bin_x'],df['Bout_x']))
    sampleRate = 16
    B1, B2, fs, delta_B, n, p = B[0], B[1], sampleRate, 0.22, 10, 98
    #%% Plot Mixed Signals
    fig, ax = plt.subplots(2,1)
    ax[0].set_title("Mixed Signals", Fontsize = '16')
    for i in range(2):
        ax[i].set_ylabel('nT', Fontsize = '12')
        ax[i].plot(samples, B[i])
        ax[i].tick_params(labelsize='8' )
        if(i != 1): plt.setp(ax[i].get_xticklabels(), visible=False)
    ax[1].set_xlabel('seconds', Fontsize = '12')


    amb_mf = REAM.gradiometry_filter(B[0], B[1], 50, .3, 10, 98)
    return

if __name__ == "__main__":
    run()