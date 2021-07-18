

#importing necessary libraries
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import sklearn 
import glob
import pydub
from pydub import AudioSegment
import time
import librosa
import noisereduce as nr
from scipy.io import wavfile
import IPython
import sounddevice as sd
from pysndfx import AudioEffectsChain
import python_speech_features
import sox
import math

#Function to convert files fo different encodings to .WAV format
count=0
for files in glob.iglob(r'C:\Users\pranj\OneDrive\Desktop\Project\train audio\*',recursive=True):
    print(files)
    count+=1
    name=files.split('\\')
    print(name)
    desti="C:\\Users\\pranj\\OneDrive\\Desktop\\Project\\train audio1\\"+name[7].split('.mp3')[0]+".wav"
    print(desti)
    sound=AudioSegment.from_mp3(files)
    sound.export(desti,format='wav')