import numpy as np
import pandas as pd

def load_michibiki_data():
    "Import the magnetometer data from the file"
    qzs_1 = np.loadtxt(r"SPACE_DATA\michibiki.dat", dtype=np.float, usecols=(0,4,5,6,7,8,9))
    B_qzs = qzs_1.T

    "Subtract the bias from the magnetometer data"
    B_qzs[1] -= 60 # MAM-S1 X-Axis
    B_qzs[2] -= 410 # MAM-S1 Y-Axis
    B_qzs[3] -= -202 # MAM-S1 Z-Axis
    B_qzs[4] -= -528 # MAM-S2 X-Axis
    B_qzs[5] -= -200 # MAM-S2 Y-Axis
    B_qzs[6] -= 478 # MAM-S2 Z-Axis

    B1 = np.vstack((B_qzs[1], B_qzs[2], B_qzs[3]))
    B2 = np.vstack((B_qzs[4], B_qzs[5], B_qzs[6]))
    michibiki = np.stack((B1, B2))
    return(michibiki)


def load_swarm_data(start = 160000, stop = 165000):
    "Import 50 Hz magnetometer residual data"
    df=pd.read_csv('SPACE_DATA\Swarm_MAGA_HR_20150317_0900.csv', sep=',',header=None)
    r = df[10]
    swarm = np.array([np.fromstring(r[i][1:-1], dtype=float, sep=' ') for i in range(1, r.shape[0])]).T[:,start:stop]
    return(swarm)