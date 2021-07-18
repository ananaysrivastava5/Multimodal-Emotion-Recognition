# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:08:57 2020

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



X1=pd.read_csv('X_new1m.csv')
X=X1.iloc[:,4:]
data=pd.read_csv('train_sent_emom.csv')
y1=data
y=data.iloc[:,1:2]

y['Emotion'].unique()
#y=y.replace(to_replace='calm',value='neutral')
#y=y.replace(to_replace='disgust',value='sad')

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
y=le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
xscale=StandardScaler()
X_train=xscale.fit_transform(X_train)
X_test=xscale.fit_transform(X_test)


from sklearn.preprocessing import MinMaxScaler
xscale=MinMaxScaler(feature_range=(0,1))
X_train=xscale.fit_transform(X_train)
X_test=xscale.fit_transform(X_test)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)

  
##basic methods of feature selection 
#features with low(zero) variance
from sklearn.feature_selection import VarianceThreshold
var = VarianceThreshold(threshold=0.1)
var = var.fit(X,y)
cols = var.get_support(indices=True)
cols
X=var.transform(X)
#X_train=var.transform(X_train)
#X_test=var.transform(X_test)

X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
X=pd.DataFrame(X)
#removing correlated features
threshold = 0.99
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
X=X.drop(columns=to_drop)
#X_train = X_train.drop(columns = to_drop)
#X_test = X_test.drop(columns = to_drop)



#using statistical method for feature selection
#using the univariatetest
#used chi2
#used f_classif
#used mutual_info_classif
#to be used Fdr,Fpr,Fwe
from sklearn.feature_selection import SelectPercentile,f_classif,mutual_info_classif,chi2
X=SelectPercentile(f_classif,percentile=80).fit_transform(X,y)
#X_train=convt.transform(X_train)
#X_test=convt.transform(X_test)




#using wrapper methods for feature selection
#using backward elimination
#Not recommended tooo slow
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestClassifier
# Estimator
estimator = RandomForestClassifier(n_estimators=10, n_jobs=-1)
# Step Forward Feature Selector
StepBackward = sfs(estimator,k_features="best",forward=False,floating=False,verbose=2,scoring='accuracy',cv=5)
StepBackward.fit(X,y)
StepBackward.subsets_

#using exhaustive elimmintion
#very expensive and not recommended but just trying as very robust
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.tree import DecisionTreeClassifier
efs=ExhaustiveFeatureSelector(DecisionTreeClassifier(),min_features=50,max_features=198,print_progress=True,scoring='accuracy',cv=5)
efs.fit(X,y)  

#using recursive methods
#rfe
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
estimator = RandomForestClassifier(n_estimators=10, n_jobs=-1)
rfe = RFE(estimator=estimator, n_features_to_select=94, step=1)
RFeatures = rfe.fit(X, y)
rfe.ranking_
cols = rfe.get_support(indices=True)
cols


#rfecv
#y1=np.ravel(y)
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
clf =DecisionTreeClassifier()
trans = RFECV(clf)
X = trans.fit_transform(X, y)

#using high-end genetic algorithm
from genetic_selection import GeneticSelectionCV
# import your preferred ml model.
from sklearn.tree import DecisionTreeClassifier
#build the model with your preferred hyperparameters.
model = DecisionTreeClassifier()
# create the GeneticSelection search with the different parameters available.
selection = GeneticSelectionCV(model,
                              cv=5,
                              scoring="accuracy",
                              max_features=198,
                              n_population=120,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=50,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              n_gen_no_change=10,
                              n_jobs=-1)
# fit the GA search to our data.
selection = selection.fit(X, y)
X=selection.transform(X)


#using feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
estimator = RandomForestClassifier(n_estimators=10)
estimator = estimator.fit(X, y)
estimator.feature_importances_
model = SelectFromModel(estimator, prefit=True)
cols = model.get_support(indices = True)
cols
X=model.transform(X)
#X_train=model.transform(X_train)
#X_test=model.transform(X_test)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
xscale=StandardScaler()
X_train=xscale.fit_transform(X_train)
X_test=xscale.transform(X_test)












from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=3000,random_state=5)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred,normalize=True))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
