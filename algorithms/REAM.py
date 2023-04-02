# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:58:17 2022

@author: alexp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift
import pandas as pd

def clean(sig, r1=1, r2=2):
    a = (r1/r2)**3
    diff = sig[0] - sig[1] # Inner - Outer
    diff = pd.Series(diff)
    
    "Find Rolling Max"
    N = 120*fs
    db = 4
    db_max = diff.rolling(N).max()
    db_min = diff.rolling(N).min()
    
    
    
