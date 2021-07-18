# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:22:19 2020

@author: pranj
"""

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

#Function implementing librosa
import librosa
feature=[]
count=0
df=pd.DataFrame([])
#formation of mapping for the dataset
files=os.listdir(r'C:\Users\pranj\OneDrive\Desktop\Project\train audio1')
dict1={}
count=0
for file in files:
    print(file)
    name=file.split('.wav')
    x=name[0].split('_')
    print(name)
    x1=list(x[0])
    x2=list(x[1])
    diano=''.join(x1[3:])
    uttno=''.join(x2[3:])
    if(diano in dict1.keys()):
        dict1[diano].append(uttno)
        count+=1
    else:
        dict1[diano]=[]
        dict1[diano].append(uttno)
        count+=1
print(count)

#Getting the feature vector using the mapping to acccess the files 
keys=list(map(int,dict1.keys()))
keys.sort()
feature=[]
count=0
for i in range(len(keys)):
    key1=str(keys[i])
    utterance=list(map(int,dict1[str(keys[i])]))
    utterance.sort()
    print(utterance)
    for j in range(len(utterance)):
        l=[]
        result=np.array([])
        location="C:\\Users\\pranj\\OneDrive\\Desktop\\Project\\train audio1\\dia"+key1+"_"+"utt"+str(utterance[j])+".wav"
        print(location)
        count+=1
        
        #y,sample_rate=librosa.load(location)
        
        #zcr=np.mean((librosa.feature.zero_crossing_rate(y=y,frame_length=sample_rate).T),axis=0)
        #rmse = np.mean((librosa.feature.rms(y=y,frame_length=sample_rate).T),axis=0)
        #spec_cent = np.mean((librosa.feature.spectral_centroid(y=y, sr=sample_rate).T),axis=0)
        #spec_bw = np.mean((librosa.feature.spectral_bandwidth(y=y, sr=sample_rate).T),axis=0)
        #rolloff = np.mean((librosa.feature.spectral_rolloff(y=y, sr=sample_rate).T),axis=0)
        #stft = np.abs(librosa.stft(y))
        #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        #mel = np.mean(librosa.feature.melspectrogram(y, sr=sample_rate).T,axis=0)
        #mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sample_rate,n_mfcc=25).T,axis=0)
        
        #result=np.hstack((result, zcr))
        #result=np.hstack((result, rmse))
        #result=np.hstack((result, spec_cent))
        #result=np.hstack((result, spec_bw))
        #result=np.hstack((result, rolloff))
        #result=np.hstack((result, chroma))
        #result=np.hstack((result, mel))
        #result=np.hstack((result,mfcc))
        #l=list(result)
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=2)
        X1 = smile.process_file(location)
        df=df.append(X1)
        if(j==2):
            break
    if(i==2):
        break
        #l=np.array(X1)
        #feature.append(l)
        

#X=np.zeros((1,len(feature[0])),dtype=float)
#for i in range(len(feature)):
#    print(i)
#    X[i]=feature[i]

#data1=pd.read_csv('X_new1.csv',header=None)
data=pd.read_csv('train_sent_emo.csv')
y=data.iloc[:,0:1]
y.describe()
y.info()
y['Emotion'].unique()

#using opensmile
import opensmile



#constructing the emotion column
#y=y.replace(to_replace='calm',value='neutral')
y=y.replace(to_replace='disgust',value='sadness')
y['Emotion'].unique()
y=np.array(y)
#y.describe()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
y=le.fit_transform(y)
y=np.reshape(y,(9988,1))
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('Emotions' ,OneHotEncoder(), [0])],remainder='passthrough')
y=ct.fit_transform(y)
df = pd.DataFrame(y.toarray())
y=df

#One-hot encoding is not necessary as we are using a random forest classification that doesnt require
#one-hot encoded data.Also it would treat one hot encoded data as mutli-label instead of multi-class.
#y=pd.get_dummies(y,columns=['EMOTION'])

#Function to split cleaned data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.2,random_state=5)

#Normalizing the Feature vector  
from sklearn.preprocessing import StandardScaler
xscale=StandardScaler()
X_train=xscale.fit_transform(X_train)
X_test=xscale.fit_transform(X_test)
#np.savetxt('X_train.csv',X_train,delimiter=",")
#np.savetxt('X_test.csv',X_test,delimiter=",")


#implementing PCA
#from sklearn.decomposition import PCA
#decomposer=PCA(.98)
#X_train=decomposer.fit_transform(X_train)
#eX_test=decomposer.transform(X_test)

#applying LDA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#decomposer=LDA(n_components=80)
#decomposer.fit(X_train,y_train)
#X_train=decomposer.transform(X_train)
#X_test=decomposer.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=3000,random_state=0,class_weight={"neutral": 4.0, "surprise": 15.0, 'fear': 15.0, 'sadness': 3.0, 'joy': 1.0,  'anger': 3.0})
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
#print(classifier.coef_)

#accuracy metrics
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred,normalize=True))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

yytest=pd.get_dummies(y_test)
yypred=pd.get_dummies(y_pred)
print(sklearn.metrics.roc_auc_score(yytest,yypred,multi_class='ovr'))

print(sklearn.metrics.balanced_accuracy_score(y_test,y_pred))

print(sklearn.metrics.log_loss(yytest,yypred))