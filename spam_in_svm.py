# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 00:27:44 2017

@author: priya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

spamdata=pd.read_csv("maindata.csv", header=0)

# building feature matrix
y=spamdata.iloc[:,57].values
X=spamdata.iloc[:,0:56].values

# splitting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)

 # feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# fitting svm model to training dataset
from sklearn.svm import  SVC
svm_model=SVC(kernel="rbf", random_state=0)
svm_model.fit(X_train,y_train)
labels=np.unique(y_train)
print(labels)
y_pred=svm_model.predict(X_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

# accuracy score
from sklearn.metrics import accuracy_score
print accuracy_score(y_test, y_pred)
