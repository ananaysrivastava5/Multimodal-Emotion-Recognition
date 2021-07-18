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
import scipy as sp
def power(y, sr):

    feature = librosa.feature.spectral_centroid(y=y, sr=sr)

    h = round(np.median(feature))*1.5
    l = round(np.median(feature))*0.1

    noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=l, slope=0.5).highshelf(gain=20.0, frequency=h, slope=0.1)#.limiter(gain=6.0)
    clean = noise(y)

    return clean


'''
    Reduction Using Spectral centroid
    -->Spectral centroid is a measure of location of featurere of mass of the sound envelope
       that directly co-relates with a brightness of sound impression. Using this fundamnetally
       we will focus to increase the brighter part and decrease the dimmer section.
    -->This will take a matrix of centroid values and return the refined or say gained matrix 
       to calling function

'''

def spectral_centroid(y, sr):

    feature = librosa.feature.spectral_centroid(y=y, sr=sr)

    h = np.max(feature)
    l = np.min(feature)

    noise = AudioEffectsChain().lowshelf(gain=-12.0, frequency=l, slope=0.5).highshelf(gain=-12.0, frequency=h, slope=0.5).limiter(gain=6.0)

    cleaned = noise(y)

    return cleaned

def spectral_centroid1(y, sr):

    feature = librosa.feature.spectral_centroid(y=y, sr=sr)

    h = np.max(feature)
    l = np.min(feature)

    noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=l, slope=0.5).highshelf(gain=-30.0, frequency=h, slope=0.5).limiter(gain=10.0)
    # less_noise = AudioEffectsChain().lowpass(frequency=threshold_h).highpass(frequency=threshold_l)
    cleaned = noise(y)


    cleaned1 = librosa.feature.spectral_centroid(y=cleaned, sr=sr)
    columns, rows = cleaned1.shape
    h1 = math.floor(rows/3*2)
    l1 = math.floor(rows/6)
    boost1 = math.floor(rows/3)

    # boost_bass = AudioEffectsChain().lowshelf(gain=20.0, frequency=boost, slope=0.8)
    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)#.lowshelf(gain=-20.0, frequency=boost_l, slope=0.8)
    clean_boosted = boost_bass(cleaned1)

    return clean_boosted


'''
--->MFCC that is Mel Frequency Ceptral Coefficients are a set of 40 different features that
    are extensively used to imitate human voice. We used MFCC to reduce noise as we are
    focusing majorly on the human voice hence, it will be a great factor to remove any other 
    sound.
--->There can be two methods to be applied ton MFCC'S. Most of the time noise stays in background
    hence, we can just lowershelf the whole component that will reduce noise values to almost NULL
    while the second method can be to highshelf each feature. Yes, it will also increase the noise
    but the main sound will rise exponentially that will help in clearly distinguishing noise to main sound.
    
'''
def mfcc_down(y, sr):

    hop_length = 512

    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum1 = []
    index = -1
    for i in mfcc:
        sum1.append(0)
        index = index + 1
        for n in i:
            sum1[index] = sum1[index] + n**2

    best_frame = sum1.index(max(sum1))
    frequency = python_speech_features.base.mel2hz(mfcc[best_frame])

    max_frequency = max(frequency)
    min_frequency = min(frequency)

    boosting = AudioEffectsChain().highshelf(frequency=min_frequency*(-.01), gain=10.0, slope=0.9)#.limiter(gain=8.0)
    y_new = boosting(y)

    return (y_new)

def mfcc_up(y, sr):

    hop_length = 512

    ## librosa
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # librosa.mel_to_hz(mfcc)

    ## mfcc
    mfcc = python_speech_features.base.mfcc(y)
    mfcc = python_speech_features.base.logfbank(y)
    mfcc = python_speech_features.base.lifter(mfcc)

    sum1 = []
    index = -1
    for i in mfcc:
        sum1.append(0)
        index = index + 1
        for n in i:
            sum1[index] = sum1[index] + n**2

    best_frame = sum1.index(max(sum1))
    frequency = python_speech_features.base.mel2hz(mfcc[best_frame])

    max_frequency = max(frequency)
    min_frequency = min(frequency)

    boosting = AudioEffectsChain().lowshelf(frequency=min_frequency*(-5), gain=75.0, slope=.9)#.highshelf(frequency=min_hz*(-1)*1.2, gain=-12.0, slope=0.5)#.limiter(gain=8.0)
    y_new = boosting(y)

    return (y_new)

'''
--->Sometimes a very vauge and simple method can be very effective. Based on this thought we use a simple method to just 
    apply a basic filter based on stastical features e.g Mean, Median. As voice sample doesn't follow 
    gussian distribution, using mean will not be that effective and hence we choose Median 
    as a filter to choose the signals.

 '''


def reduce_noise_median(y, sr):
    y = sp.signal.medfilt(y,3)
    return (y)


'''
--->Silences in an audio may occur anywhere, they might be at beginning or at end or sometimes
    at middle but these silences act as outliers and they mostly have 0 in all feature values,
    as well they increase the length of feature vector.It is necessary to remove silences from 
    audio but it is important to dintinguish between a wanted and a unwanted silences.
--->For this, we used Trim function of librosa library that convert audio into volumn envelope
    and to dintinguish between wanted and unwanted silences, we used a minimmum frame length of 2
    and a thresshold decibal value.
'''
def remove_silence(y):
    trim, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=512)
    length_removed = librosa.get_duration(y) - librosa.get_duration(trim)

    return trim, length_removed






data,sr=librosa.load(r"C:\Users\pranj\OneDrive\Desktop\Project\test.wav",sr=5000)
reduced_noise=mfcc_down(data,sr)
print(data)
IPython.display.Audio(data=reduced_noise, rate=sr)
sd.play(reduced_noise, sr)
status = sd.wait()