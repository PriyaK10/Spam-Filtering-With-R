# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:55:56 2017

@author: 1542283
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})


np.random.seed(500)
k = 3
# centroids[i] = [x, y]

df1=df.sample(n=3)

cen={
     i+1:[df1.iloc[i,0],df1.iloc[i,1]]
     for i in range(k)
     }

#plotting centroids
fig=plt.figure(figsize=(5,5))
plt.scatter(df.iloc[:,0], df.iloc[:,1], color='k')

colmap = {1: 'red', 2: 'green', 3: 'blue'}

for i in cen.keys():
    plt.scatter(cen[i][0],cen[i][1], color=colmap[i])
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()





#Assignment Stage

def assignment(X, centroids):
    for i in centroids.keys():
       
        X['dist_from_Cluster_{}'.format(i)] = (
            np.sqrt(
                (X.iloc[:,0] - centroids[i][0]) ** 2
                + (X.iloc[:,1] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['dist_from_Cluster_{}'.format(i) for i in centroids.keys()]
    X['assigned_to_Cluster'] = X.loc[:, centroid_distance_cols].idxmin(axis=1)
    X['assigned_to_Cluster'] = X['assigned_to_Cluster'].map(lambda x: int(x.lstrip('dist_from_Cluster_')))
    X['color'] = X['assigned_to_Cluster'].map(lambda x: colmap[x])
    return X
            
            
            
df = assignment(df, cen)
print(df.head())


fig = plt.figure(figsize=(5, 5))
plt.scatter(df.iloc[:,0], df.iloc[:,1], color=df['color'], alpha=0.5, edgecolor='k')
for i in cen.keys():
    plt.scatter(*cen[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()

#Update centroid
old_centroids=cen

def update(k):
    for i in cen.keys():
        cen[i][0] = np.mean(df[df['assigned_to_Cluster'] == i].iloc[:,0])
        cen[i][1] = np.mean(df[df['assigned_to_Cluster'] == i].iloc[:,1])
    return k

cen = update(cen)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df.iloc[:,0], df.iloc[:,1], color=df['color'], alpha=0.5, edgecolor='k')
for i in cen.keys():
    plt.scatter(*cen[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)


while True:
    closest_centroids = df['assigned_to_Cluster'].copy(deep=True)
    cen = update(cen)
    df = assignment(df, cen)
    
    if closest_centroids.equals(df['assigned_to_Cluster']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df.iloc[:,0], df.iloc[:,1], color=df['color'], alpha=0.5, edgecolor='k')
for i in cen.keys():
    plt.scatter(*cen[i], color=colmap[i])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()





