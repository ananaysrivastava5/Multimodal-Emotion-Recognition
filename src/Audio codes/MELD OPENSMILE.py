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
import csv

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
        smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        num_channels=2)
        X1 = smile.process_file(location)
        df=df.append(X1)
        
        
df.to_csv('X_new1.csv',sep=',')

data=pd.read_csv('train_sent_emo.csv')
y=data.iloc[:,0:1]
y.describe()
y.info()
y['Emotion'].unique()




y=y.replace(to_replace='disgust',value='sadness')
y['Emotion'].unique()
y=np.array(y)
#y.describe()
df=pd.read_csv('X_new1.csv')

print(y['Emotion'].value_counts(dropna=False))
#Dropping rows to balance dataset
print(y.shape)
counter=0
for i in range(y.shape[0]-1,-1,-1):
    print(i)
    if(counter>3000):
        break
    if(y['Emotion'].loc[i]=='neutral'):
        y=y.drop(y.index[i])
        df=df.drop(df.index[i])
        counter+=1
    

df.to_csv('X_new1m.csv',sep=',')
y.to_csv('train_sent_emom.csv',sep=',')
print(y['Emotion'].value_counts(dropna=False))
#label and one hot encoding

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
y=le.fit_transform(y)


y=np.reshape(y,(9988,1))
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([('Emotions' ,OneHotEncoder(), [0])],remainder='passthrough')
y=ct.fit_transform(y)
df = pd.DataFrame(y.toarray())
y=df



#Function to split cleaned data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df,y,test_size=0.2,random_state=5)

#Normalizing the Feature vector  
from sklearn.preprocessing import StandardScaler
xscale=StandardScaler()
X_train=xscale.fit_transform(X_train)
X_test=xscale.fit_transform(X_test)



#implementing PCA
from sklearn.decomposition import PCA
decomposer=PCA(.98)
X_train=decomposer.fit_transform(X_train)
X_test=decomposer.transform(X_test)




#model training
from sklearn.svm import SVC
classifier=SVC()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(y_test)

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