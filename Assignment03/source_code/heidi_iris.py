# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 17:03:55 2016

@author: Ayushi
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import math
import operator
from scipy.spatial import distance

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

#approach1 : nearest to centroid
#ordering each class's points based on distance from centroid
def getDistances(trainingSet, testInstance):
    distances=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    return distances

#approach2 : connected distance between points
def getnearestPoint(trainingSet, testInstance,done):
    length=len(testInstance)-1
    pos=0
    min_dist=100
    point=()
    for x in range(1,len(trainingSet)):
        if(done[x]==False):
            pos=x
            min_dist=euclideanDistance(testInstance,trainingSet[x],length)
            point=trainingSet[x]
            break
    for x in range(pos,len(trainingSet)):
        if(done[x]==False):
            dist=euclideanDistance(testInstance,trainingSet[x],length)          
            if(dist<min_dist):
                min_dist=dist
                point=trainingSet[x]
                pos=x
    return [min_dist,point,pos]

#apporach3 : minimum spanning tree
def minSpanningTree(trainingSet,testInstance):
    dists = distance.cdist(trainingSet,trainingSet, 'euclidean')
    mst=minimum_spanning_tree(dists)
    mst=mst.toarray().astype(float)
    done=[False]*len(trainingSet)
    point=getnearestPoint(trainingSet,testInstance,done)
    distances=[]
    distances.append((point[1],point[0])) #(point, distance)
    done[point[2]]=True
    stack = [[point[1],point[2]]] #(point,index,distance)
    while len(stack)>0:
        point=stack.pop()
        temp=list(np.nonzero(mst[point[1]]))[0]
        temp1=list(np.nonzero(mst[:,point[1]]))[0]
        temp=list(temp)+list(temp1)    
        for i in temp:
            if done[i]==False:
                k=[trainingSet[i],i]
                stack.extend([k])
                distances.append((trainingSet[i],-1.0))
                done[i]=True
    return distances
    
def getConnectedDistances(trainingSet,point):
    done=[False]*len(trainingSet)
    distances=[]
    while(not all(done)):
        tpl=getnearestPoint(trainingSet,point,done)
        done[tpl[2]]=True
        distances.append((tpl[1],tpl[0]))
        point=tpl[1]
    return distances       
    

def onclick(event):
    print('------------------')
    print('x_coordinate: %f, y_coordinate: %f' %(event.xdata, event.ydata))
    x=sorted_data.iloc[int(event.xdata),0:3].values
    y=sorted_data.iloc[int(event.ydata),0:3].values
    print ("%s %s" %(x,y))
    print (bin(heidi_matrix[int(event.xdata),int(event.ydata)]))
    
data=pd.read_csv(filepath_or_buffer='./dataset/iris/iris.data',sep=',',header=None)
row=len(data.index)
col=len(data.columns)
classlabel=data.copy()
classlabel=classlabel.iloc[:,-1]
data=data.iloc[:,0:(col-1)]

k_means=KMeans(n_clusters=3)
k_means.fit(data);
print(k_means.labels_)

fig,ax=plt.subplots()

fig=plt.figure(1,figsize=(4,3))
plt.clf()
ax = Axes3D(fig)
plt.cla()
ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],c=k_means.labels_)
plt.show()

centroids=k_means.cluster_centers_

data['class']=k_means.labels_

class1=data[data.iloc[:,-1]==0]
class2=data[data.iloc[:,-1]==1]
class3=data[data.iloc[:,-1]==2]


#removing classlabel column
class1=class1.iloc[:,0:4]
class2=class2.iloc[:,0:4]
class3=class3.iloc[:,0:4]



#dist1=getDistances(class1.values,centroids[0])
#dist1=getConnectedDistances(class1.values,centroids[0])
dist1=minSpanningTree(class1.values,centroids[0])


sorted_class1=[]
for (a,b) in dist1:
    sorted_class1+=[a]
sorted_class1=pd.DataFrame(sorted_class1)    
sorted_class1['class']=0    

#dist2=getDistances(class2.values,centroids[1])
#dist2=getConnectedDistances(class2.values,centroids[1])
dist2=minSpanningTree(class2.values,centroids[1])

sorted_class2=[]
for (a,b) in dist2:
    sorted_class2+=[a]
sorted_class2=pd.DataFrame(sorted_class2)
sorted_class2['class']=1    
allsorted_data=[sorted_class1, sorted_class2]
sorted_data = pd.concat([sorted_class1, sorted_class2])

#dist3=getDistances(class3.values,centroids[2])
#dist3=getConnectedDistances(class3.values,centroids[2])
dist3=minSpanningTree(class3.values,centroids[2])

sorted_class3=[]
for (a,b) in dist3:
    sorted_class3+=[a]
sorted_class3=pd.DataFrame(sorted_class3)
sorted_class3['class']=1    

sorted_data = pd.concat([sorted_class1, sorted_class2,sorted_class3])


k=10
heidi_matrix=np.zeros(shape=(row,row),dtype=np.uint64)
max_count=int(math.pow(2,col))
allsubspaces=range(1,max_count)
f=lambda a:sorted(a,key=lambda x:sum(int(d)for d in bin(x)[2:]))
allsubspaces=f(allsubspaces)
frmt=str(col)+'b'
factor=1
for i in allsubspaces:
    bin_value=str(format(i,frmt))
    bin_value=bin_value[::-1]
    subspace_col=[index for index,value in enumerate(bin_value) if value=='1']
    print ("%d : %s : '%s'" %(i,subspace_col,bin_value[::-1]))
    subspace=sorted_data.iloc[:,subspace_col]    
    np_subspace=subspace.values
    nbrs=NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(np_subspace)
    temp=nbrs.kneighbors_graph(np_subspace).toarray()
    temp=temp.astype(np.uint64)
    heidi_matrix=heidi_matrix + temp*factor
    factor=factor*2
fig = figure()
ax1 = fig.add_subplot(111)
col1 = ax1.imshow(heidi_matrix, interpolation='nearest',picker=True)
cid = fig.canvas.mpl_connect('button_press_event', onclick)




