# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:54:39 2016

@author: Ayushi
"""



import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
import operator
import numpy as np

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getDistances(trainingSet, testInstance):
    distances=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance,trainingSet[x],length)
        q=trainingSet[x][0]-testInstance[0]
        if(q>0) :
            q=int(1)
        else :
            q=int(-1)
        distances.append((trainingSet[x],dist,q))
    distances.sort(key=operator.itemgetter(1))
    return distances

colorlist=ListedColormap(['#FF0000','#00FF00','#0000FF'])
y=[1,1, 1, 1, 1, 1, 1, 1, 1, 1]
N=10
a = np.random.rand(N,1)*10
ar = np.ones((N,2))*5
ar[:,:-1] = a
#ar=temp5
distances=getDistances(ar,ar[0])
neighbors = []
k=6; # k+1 of Knn (here its 5 knn) (1 extra added because point itself lies within dataset)
for x in range(k):
	neighbors.append(distances[x][0])
nbrs=np.array(neighbors)

cl=[i in nbrs[:,0] for i in ar[:,0]]
fig,ax=plt.subplots()
plt.axis([0, 10, 0, 10])
plt.scatter(ar[:,0],ar[:,1],c=cl,cmap=colorlist)
ax.annotate('p', xy=(ar[0][0],ar[0][1]),arrowprops=dict(facecolor='black', shrink=0.05))
#attempt1
#epsilon=(distances[k][1]-distances[k-1][1])/2 #dk+1-dk

#attempt2
#p_k1=distances[k][0]
#p_distances=getDistances(ar,p_k1)
#flag=0
#for j in p_distances:
#    row=0
#    for i in neighbors:
#        if j[0][0]==i[0]:
#            nearestnbr=distances[row]
#            flag=1
#            break
#        row+=1
#    if flag==1:
#        break
#epsilon=distances[k][1]-nearestnbr[1]

#attempt3
#epsilon=min(x1,x2)
positive_outlier=([10,5],10-ar[0][0],int(1))
negative_outlier=([0,5],ar[0][0],int(1))
x1=0.0
y1=0.0
for x in range(k):
    if(distances[x][2]==int(-1)):
        negative=distances[x]
    else :
        positive=distances[x]
for x in range(len(distances)-1,k-1,-1):
    if(distances[x][2]==int(-1)):
        negative_outlier=distances[x]
    else :
        positive_outlier=distances[x]
if(positive_outlier==()):
    epsilon=  negative_outlier
elif(negative_outlier==()):
    epsilon=positive_outlier
else:
    x2=(positive_outlier[0][0]+negative[0][0])/2 #midpoint calculation
    x1=(negative_outlier[0][0]+positive[0][0])/2 #midpoint calculation
    if((ar[0][0]-x1)>(x2-ar[0][0])):
        epsilon=x2-ar[0][0]
    else:
        epsilon=ar[0][0]-x1
        
cir=plt.Circle(ar[0],distances[k-1][1],facecolor='none')
ax.add_artist(cir)
extreem_x1=np.array([x1,5], dtype=np.float64)
extreem_x2=np.array([x2,5], dtype=np.float64)
cir1=plt.Circle(extreem_x1,euclideanDistance(extreem_x1,positive[0],1),facecolor='none')
cir2=plt.Circle(extreem_x2,euclideanDistance(extreem_x2,negative[0],1),facecolor='none')
ax.add_artist(cir1)
ax.add_artist(cir2)
ax.annotate('x1', xy=extreem_x1,arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('x2', xy=extreem_x2,arrowprops=dict(facecolor='black', shrink=0.05))
