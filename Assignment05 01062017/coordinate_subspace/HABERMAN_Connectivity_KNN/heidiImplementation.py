# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 17:35:28 2016

@author: Ayushi
"""

import pandas as pd
import numpy as np
import math
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#approach 1: ordering based on distance from centroid
def getDistances(allData,centroid):
    row,col=allData.shape
    allData['distance']=-1
    for i in range(row):
        dist=distance.euclidean(allData.iloc[i,0:col-1],centroid)
        allData.iloc[i,-1]=dist
    allData=allData.sort_values(['distance'],ascending=['False'])
    return allData

#approach2 : connected distance between points
def closest_node(allData, node):
    row,col=allData.shape
    dist = np.sqrt(np.sum((allData.iloc[:,0:col-3] - node)**2, axis=1))
    index=np.argmin(dist[~allData.loc[:,'done']])
    return index    

def getConnectedDistances(trainingSet,point):
    row,col=trainingSet.shape    
    trainingSet['done']=False
    trainingSet['pos']=-1
    count=int(0)
    while(not all(trainingSet.loc[:,'done'])):
        rownum=closest_node(trainingSet.copy(),point)
        trainingSet.loc[rownum,'done']=True
        trainingSet.loc[rownum,'pos']=count
        count+=1
        point=trainingSet.loc[rownum,:].copy()
        point=point.iloc[0:col-1]

    trainingSet=trainingSet.sort_values(['pos'],ascending=['False'])
    #print (trainingSet)
    return trainingSet.iloc[:,0:col]
 
#apporach3 : minimum spanning tree
def minSpanningTree(trainingSet,testInstance):
    row,col=trainingSet.shape    
    dists = distance.cdist(trainingSet.iloc[:,0:col-1],trainingSet.iloc[:,0:col-1], 'euclidean')
    mst=minimum_spanning_tree(dists)
    trainingSet['done']=False
    trainingSet['pos']=-1    
    count=0
    mst=mst.toarray().astype(float)
    rownum=closest_node(trainingSet,testInstance)
    point=trainingSet.loc[rownum,:].copy()  
    point=point.iloc[0:col-1]
    trainingSet.loc[rownum,'done']=True
    trainingSet.loc[rownum,'pos']=count#    distances=[]    
    stack = [trainingSet.index.get_loc(rownum)] #(point,index,distance) # serial number saved
    count+=1
    while len(stack)>0:
        rownum=stack.pop()
        rownum_orig=trainingSet.index.get_values()[rownum] #index value saved in orig
        point=trainingSet.loc[rownum_orig,:].copy()  
        point=point.iloc[0:col-1]

        temp=list(np.nonzero(mst[rownum]))[0]
        temp1=list(np.nonzero(mst[:,rownum]))[0]
        temp=list(temp)+list(temp1)    
        for i in temp:
            k=trainingSet.index.get_values()[i]
            if trainingSet.loc[k,'done']==False:
                stack.extend([i])
                trainingSet.loc[k,'done']=True
                trainingSet.loc[k,'pos']=count#    distances=[]
                count+=1

    trainingSet=trainingSet.sort_values(['pos'],ascending=['False'])
    return trainingSet.iloc[:,0:col]
    
 
def onclick(event):
    print('------------------')
    print('x_coordinate: %f, y_coordinate: %f' %(event.xdata, event.ydata))
    x=sorted_data.iloc[int(event.xdata),0:col-1].values
    y=sorted_data.iloc[int(event.ydata),0:col-1].values
    print ("%s %s" %(x,y))
    print (bin(heidi_matrix[int(event.xdata),int(event.ydata)]))
    
if __name__=='__main__':    

    inputData=pd.read_csv(filepath_or_buffer='./dataset/haberman/haberman.data',sep=',',header=None)    
    row=len(inputData.index)
    col=len(inputData.columns)
    feature_vector=inputData.iloc[:,0:col-1]
    classLabel_given=inputData.iloc[:,col-1:col]
    n_of_classes=set('1')
    #n_of_classes=set(classLabel_given.iloc[:,0]) # set of all unique classes given in dataset
    kmeans=KMeans(n_clusters=len(n_of_classes))
    classLabel_predicted=kmeans.fit_predict(feature_vector)

    centroids=kmeans.cluster_centers_
    feature_vector.loc[:,'classLabel']=classLabel_predicted
    #print (allData)
    sorted_data=pd.DataFrame()
    for i in set(classLabel_predicted):
        #temp=getDistances(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
        temp=getConnectedDistances(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
        #temp=minSpanningTree(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
        sorted_data=pd.concat([sorted_data,temp])
    print (sorted_data)
            
    k=50 #1-NN
    heidi_matrix=np.zeros(shape=(row,row),dtype=np.uint64)
    max_count=int(math.pow(2,col-1))
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
        #print(temp*factor)
        heidi_matrix=heidi_matrix + temp*factor
        factor=factor*2
#    fig = plt.figure(2)
#    ax1 = fig.add_subplot(111)
#    col1 = ax1.imshow(heidi_matrix, interpolation='nearest',picker=True)
#    cid = fig.canvas.mpl_connect('button_press_event', onclick)
#    plt.show()
    #heidi_matrix=pd.DataFrame(data=heidi_matrix,index=sorted_data.index,columns=sorted_data.index)            
    
    max_count=max_count-1  
    r=int(max_count/3)
    g=int(max_count/3)
    b=max_count-r-g
    
    x=heidi_matrix>>(max_count-r)
    y=(heidi_matrix & ((pow(2,g)-1)<<b)) >> b
    z=(heidi_matrix & (pow(2,b)-1))
#    x=x.astype(np.uint8)
#    y=y.astype(np.uint8)
#    z=z.astype(np.uint8)
    cluster_matrix=[]
    for i in range(0,row):
        for j in range(0,row):
            cluster_matrix.append([i,j,x[i][j],y[i][j],z[i][j]]) 
            
    cluster_matrix=np.array(cluster_matrix)
    #cluster_matrix=cluster_matrix.T
    db = DBSCAN(eps=10, min_samples=10).fit(cluster_matrix)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    heidi_img=np.dstack((x,y,z))
    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    col1 = ax1.imshow(heidi_img, interpolation='nearest',picker=True)
    plt.show()
    
    print(set(labels))
