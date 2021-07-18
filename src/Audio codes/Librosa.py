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
import csv
#Function implementing librosa
import librosa
feature=[]

for files in glob.iglob(r'C:\Users\pranj\OneDrive\Desktop\Project\archive\*',recursive=True):
    path=files+'\*.wav'
    print(path)
    for filename in glob.iglob(path):
        l1=[]
        result=np.array([])
        print(filename)
        
        y,sample_rate=librosa.load(filename)
    
        zcr=np.mean((librosa.feature.zero_crossing_rate(y=y,frame_length=sample_rate).T),axis=0)
        rmse = np.mean((librosa.feature.rms(y=y,frame_length=sample_rate).T),axis=0)
        spec_cent = np.mean((librosa.feature.spectral_centroid(y=y, sr=sample_rate).T),axis=0)
        spec_bw = np.mean((librosa.feature.spectral_bandwidth(y=y, sr=sample_rate).T),axis=0)
        rolloff = np.mean((librosa.feature.spectral_rolloff(y=y, sr=sample_rate).T),axis=0)
        stft = np.abs(librosa.stft(y))
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y, sr=sample_rate).T,axis=0)
        mfcc=np.mean(librosa.feature.mfcc(y=y,sr=sample_rate,n_mfcc=40).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y),sr=sample_rate).T,axis=0)
        
        result=np.hstack((result, zcr))
        result=np.hstack((result, rmse))
        result=np.hstack((result, spec_cent))
        result=np.hstack((result, spec_bw))
        result=np.hstack((result, rolloff))
        result=np.hstack((result, mfcc))
        result=np.hstack((result, chroma))
        result=np.hstack((result, mel))
        result=np.hstack((result,contrast))
        result=np.hstack((result,tonnetz))
        
        l1=list(result)
        feature.append(l1)
        #print(result.shape)
        #time.sleep(10)
#print(len(feature[0]))
X=np.zeros((1440,len(feature[0])),dtype=float)
for i in range(len(feature)):
    print(i)
    X[i]=feature[i]
with open('Ravdess_X.csv','w',newline='') as file:
    writer=csv.writer(file)
    writer.writerows(X)
data=pd.read_csv('Ravdess.csv')
y=data.iloc[:,2:3]

#constructing the emotion column
y=y.replace(to_replace='calm',value='neutral')
y=y.replace(to_replace='disgust',value='sad')
#y.describe()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

#One-hot encoding is not necessary as we are using a random forest classification that doesnt require
#one-hot encoded data.Also it would treat one hot encoded data as mutli-label instead of multi-class.
#y=pd.get_dummies(y,columns=['EMOTION'])

#Function to split cleaned data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

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
#X_test=decomposer.transform(X_test)

#applying LDA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#decomposer=LDA(n_components=80)
#decomposer.fit(X_train,y_train)
#X_train=decomposer.transform(X_train)
#X_test=decomposer.transform(X_test)


#trial model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=3000,random_state=5)
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



