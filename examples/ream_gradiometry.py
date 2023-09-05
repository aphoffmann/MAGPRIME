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
from algorithms import REAM
from algorithms import PiCoG

def run():
    df=pd.read_csv('SPACE_DATA\mstac2e2esupp1.csv', sep=',',header=0)
    samples = np.arange(0, len(df['Bin_x']))/16
    B = df[['Bin_x', 'Bin_y', 'Bin_z', 'Bout_x', 'Bout_y', 'Bout_z']].to_numpy().T
    B = B.reshape(2, 3, -1)
    sampleRate = 16

    REAM.p = 98
    REAM.delta_B = 0.1
    REAM.n = 10
    #%% Plot Mixed Signals
    fig, ax = plt.subplots(2,1)
    ax[0].set_title("Mixed Signals", fontsize = '16')
    for i in range(2):
        ax[i].set_ylabel('nT', fontsize = '12')
        ax[i].plot(samples, B[i,0,:])
        ax[i].tick_params(labelsize='8' )
        if(i != 1): plt.setp(ax[i].get_xticklabels(), visible=False)
    ax[1].set_xlabel('seconds', fontsize = '12')


    amb_mf = REAM.clean(B)
    
    return

if __name__ == "__main__":
    run()