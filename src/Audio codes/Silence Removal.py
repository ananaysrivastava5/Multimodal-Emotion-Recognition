import warnings
warnings.filterwarnings('ignore')

#importing necessary libraries
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import sklearn 
import glob

def wrapper(y,sr,threshold):
    mask=[]
    y1=pd.series(y).apply(np.abs)
    y2=y1.rolling(window=int(sr/10),min_periods=1).mean()
    for mean in y2:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(y[mask])
