# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:17:50 2017

@author: 1542283
"""

from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn import datasets 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(100)
iris = datasets.load_iris()
X = iris.data
df=pd.DataFrame(X)

feature_names = iris.feature_names
y = iris.target
target_names = iris.target_names
# examine the data# plot the data in 2D

k = 3
# centroids[i] = [x, y]

df1=df.sample(n=k)

cen={
     i+1:[df1.iloc[i,0],df1.iloc[i,1],df1.iloc[i,2],df1.iloc[i,3]]
     for i in range(k)
     }

# plotting data in 2D SEPAL
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 1, 1)
II = (y==0)
ax.scatter(X[II,0], X[II, 1], color='blue')
II = (y==1)
ax.scatter(X[II,0], X[II, 1], color='red')
II = (y==2)
ax.scatter(X[II,0], X[II, 1], color='green')
ax.set_title('sepal')
ax.set_xlabel('length')
ax.set_ylabel('width')

#PETAL DATA
ax = fig.add_subplot(2, 1, 2)
II = (y==0)
ax.scatter(X[II,2], X[II, 3], color='blue')
II = (y==1)
ax.scatter(X[II,2], X[II, 3], color='red')
II = (y==2)
ax.scatter(X[II,2], X[II, 3], color='green')
ax.set_title('petal')
ax.set_xlabel('length')
ax.set_ylabel('width')
fig.show()

#PLOT DATA in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
y1 = np.choose(y, [1, 2, 0])
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y1, edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

flower_name_and_label = [('Setosa', 0),
                         ('Versicolor', 1),
                         ('Virginica', 2)]
for name, label in flower_name_and_label:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
fig.show()

#Assignment Stage
colmap = {1: 'red', 2: 'green', 3: 'blue'}

def assignment(X, centroids):
    for i in centroids.keys():
       
        X['dist_from_Cluster_{}'.format(i)] = (
            np.sqrt(
                (X.iloc[:,0] - centroids[i][0]) ** 2
                + (X.iloc[:,2] - centroids[i][2]) ** 2
                + (X.iloc[:,3] - centroids[i][3]) ** 2
            )
        )
    centroid_distance_cols = ['dist_from_Cluster_{}'.format(i) for i in centroids.keys()]
    X['assigned_to_Cluster'] = X.loc[:, centroid_distance_cols].idxmin(axis=1)
    X['assigned_to_Cluster'] = X['assigned_to_Cluster'].map(lambda x: int(x.lstrip('dist_from_Cluster_')))
    X['color'] = X['assigned_to_Cluster'].map(lambda x: colmap[x])
    return X
            
            
        
df = assignment(df, cen)
print(df.head())


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
y1 = np.choose(y, [1, 2, 0])
ax.scatter(df.iloc[:, 3], df.iloc[:, 0], df.iloc[:, 2], c=y1, edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()

#Update centroid
old_centroids=cen

def update(k):
    for i in cen.keys():
        cen[i][0] = np.mean(df[df['assigned_to_Cluster'] == i].iloc[:,0])
        cen[i][1] = np.mean(df[df['assigned_to_Cluster'] == i].iloc[:,1])
        cen[i][2] = np.mean(df[df['assigned_to_Cluster'] == i].iloc[:,2])
    return k

cen = update(cen)


while True:
    closest_centroids = df['assigned_to_Cluster'].copy(deep=True)
    cen = update(cen)
    df = assignment(df, cen)
   
    if closest_centroids.equals(df['assigned_to_Cluster']):
        break
    
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
y1 = np.choose(y, [1, 2, 0])
ax.scatter(df.iloc[:, 3], df.iloc[:, 0], df.iloc[:, 2], c=y1, edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()

