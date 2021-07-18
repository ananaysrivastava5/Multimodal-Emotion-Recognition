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
import envelope
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
    
X=pd.read_csv('Ravdess_X.csv',header=None)
data=pd.read_csv('Ravdess.csv')
y=data.iloc[:,2:3]

#constructing the emotion column
y=y.replace(to_replace='calm',value='neutral')
y=y.replace(to_replace='disgust',value='sad')
#y.describe()


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

#y=le.inverse_transform(y)


#Function to split cleaned data into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)

#Normalizing the Feature vector  
from sklearn.preprocessing import StandardScaler
xscale=StandardScaler()
X_train=xscale.fit_transform(X_train)
X_test=xscale.fit_transform(X_test)
#np.savetxt('X_train.csv',X_train,delimiter=",")
#np.savetxt('X_test.csv',X_test,delimiter=",")


#feature selection
from sklearn.feature_selection import VarianceThreshold
var = VarianceThreshold(threshold=0.05)
var = var.fit(X_train,y_train)
cols = var.get_support(indices=True)
cols
X_train=var.transform(X_train)
X_test=var.transform(X_test)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
X=pd.DataFrame(X)
#removing correlated features
threshold = 0.90
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#X=X.drop(columns=to_drop)
X_train=X_train.drop(columns=to_drop)
X_test=X_test.drop(columns=to_drop)

from sklearn.feature_selection import SelectPercentile,f_classif,mutual_info_classif,chi2
selector=SelectPercentile(f_classif,percentile=80).fit(X_train,y_train)
X_train=selector.transform(X_train)
X_test=selector.transform(X_test)

from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
clf =DecisionTreeClassifier()
trans = RFECV(clf,n_jobs=-1)
trans.fit(X_train, y_train)
X_train=trans.transform(X_train)
X_test=trans.transform(X_test)

#implementing PCA
from sklearn.decomposition import PCA
decomposer=PCA(.98)
X_train=decomposer.fit_transform(X_train)
X_test=decomposer.transform(X_test)




#trial model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid_search = {'penalty': ['l2'],
               'C': [1,2,3,5,10],
               'class_weight': ['balanced'],
               'max_iter': [1000, 500, 3000],
               'random_state': [0, 5,100],
               'solver': ['newton-cg','lbfgs']}

clf = LogisticRegression()
model = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 4, verbose= 5, n_jobs = -1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
#accuracy metrics


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

grid_search = {'n_neighbors': [5,10,15,20],
               'weights': ['uniform','distance'],
               'p': [1,2,5],
               }

clf =   KNeighborsClassifier()
model = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 4, verbose= 5, n_jobs = -1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_search = {'C': [1,2,3,5],
               'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
               'random_state': [0,5,100],
               }

clf =   SVC()
model = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 4, verbose= 5, n_jobs = -1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)



from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

grid_search = {
               }

clf =   GaussianNB()
model = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 4, verbose= 5, n_jobs = -1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
grid_search = {'criterion': ['entropy', 'gini'],
               'max_depth': [2],
               'max_features': ['auto', 'sqrt'],
               'min_samples_leaf': [4, 6, 8],
               'min_samples_split': [5, 7,10],
               'n_estimators': [20]}

clf = RandomForestClassifier()
model = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 4, verbose= 5, n_jobs = -1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
grid_search = {}

clf = GradientBoostingClassifier()
model = GridSearchCV(estimator = clf, param_grid = grid_search, 
                               cv = 4, verbose= 5, n_jobs = -1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)




y_pred=le.inverse_transform(y_pred)
y_test=le.inverse_transform(y_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred,normalize=True))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))





