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
import pydub
from pydub import AudioSegment
import time
import opensmile
import csv

#Function implementing librosa
import librosa
import speech_recognition as sr
filename=r'C:\Users\pranj\Downloads\IEMOCAP_full_release_withoutVideos\IEMOCAP_full_release\Session1\dialog\wav\Ses01F_impro01.wav'
r = sr.Recognizer()
with sr.AudioFile(filename) as source:
    # listen for the data (load audio to memory)
    audio_data = r.record(source)
    print(audio_data)
    # recognize (convert from speech to text)
    text = r.recognize_google(audio_data,show_all=True)
    print(text)