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



#y,sr=librosa.load(r"C:\Users\pranj\OneDrive\Desktop\Project\72843_lonemonk_approx-800-laughter-only-1.wav")
my,sr=librosa.load(r"C:\Users\pranj\Downloads\IEMOCAP_full_release_withoutVideos\IEMOCAP_full_release\Session1\sentences\wav\Ses01F_impro01\Ses01F_impro01_F000.wav")
reduced_noise = nr.reduce_noise(audio_clip=my, noise_clip=my, verbose=True,prop_decrease=0.8)
print(IPython.display.Audio(data=my, rate=sr))
sd.play(my, sr)
status = sd.wait()