# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:55:00 2017

@author: Ayushi
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import math
import operator
from mpl_toolkits.mplot3d import Axes3D

#approach1 : nearest to centroid
#ordering each class's points based on distance from centroid
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
    trainingSet.loc[rownum,'pos']=count 
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

if __name__=='__main__':
    inputData=pd.read_csv(filepath_or_buffer='./dataset/iris/iris.data',sep=',',header=None) # data is nXm  n :no of datapoints  m :no. of features  
    row=len(inputData.index)   # total number of datapoints/rows in given dataset
    col=len(inputData.columns) #total number of coulmns/features in given dataset
    
    feature_vector=inputData.iloc[:,0:col-1]
    classLabel_given=inputData.iloc[:,col-1:col]
    
    n_of_classes=set(classLabel_given.iloc[:,0]) # set of all unique classes given in dataset
    
    kmeans=KMeans(n_clusters=len(n_of_classes))
    classLabel_predicted=kmeans.fit_predict(feature_vector)
    centroids=kmeans.cluster_centers_
    feature_vector.loc[:,'classLabel']=classLabel_predicted
        
    sorted_data=pd.DataFrame()
    for i in set(classLabel_predicted):
        #temp=getDistances(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
        #temp=getConnectedDistances(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
        temp=minSpanningTree(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
        sorted_data=pd.concat([sorted_data,temp])        
        
    training_data=sorted_data.iloc[:,0:col-1].values
    mean_vec = np.mean(training_data, axis=0)
    cov_mat = np.cov(training_data.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    eig_vals,eig_vecs=zip(*eig_pairs)
    eig_vecs=np.array(eig_vecs)    
    tot = sum(eig_vals)
    var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    X=(training_data - mean_vec).T
    output=eig_vecs[:,:].dot(X)
    output=output.T

#-----------actual overlap in suspace matrix
    overlap_matrix=np.zeros(shape=(row,row),dtype=np.uint64)
    max_count=int(math.pow(2,col-1))
    allsubspaces=range(max_count-1,0,-1)
    frmt=str(col)+'b'
    factor=1

    for i in allsubspaces:
        bin_value=str(format(i,frmt))
        bin_value=bin_value[::-1]
        subspace_col=[index for index,value in enumerate(bin_value) if value=='1']
        subspace=output[:,subspace_col]
        #np_subpace=subspace.values
        dists = distance.cdist(subspace,subspace, 'euclidean')
        dists=dists==0
        dists=dists.astype(int)
        overlap_matrix=overlap_matrix + dists*factor
        factor=factor*2
        #break

    
    
    temp_array=[]
    k=10
    heidi_matrix=np.zeros(shape=(row,row),dtype=np.uint64)

    #allsubspaces=range(max_count-1-7,0,-1)
    frmt=str(col)+'b'
    factor=1    
    
    for i in allsubspaces:
        bin_value=str(format(i,frmt))
        bin_value=bin_value[::-1]
        subspace_col=[col-2-index for index,value in enumerate(bin_value) if value=='1']
        print ("%d : %s : '%s'" %(i,subspace_col,bin_value[::-1]))
        np_subspace=output[:,subspace_col] 
        
        #modified_knn(k,np_subspace)
        dists = distance.cdist(subspace,subspace, 'euclidean')
        s = np.argsort(dists, axis=1)
        #s=s[:,0:k+1]
        T=np.zeros(shape=(row,row),dtype=np.uint64)
        for i in range(row):
            count=0
            for j in s[i] :
                if(overlap_matrix[i][j]!=0):
                    T[i][j]=1
                    count+=1
                    if(count>=k) :
                        break
            if(count>=k):
                continue
            for j in s[i] :
                if(T[i][j]==0):
                    T[i][j]=1
                    count+=1
                    if(count>=k) :
                        break
                    
                    
                    
        heidi_matrix=heidi_matrix + T*factor        
        factor=factor*2
        
        
#        nbrs=NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(np_subspace)
#        temp=nbrs.kneighbors_graph(np_subspace).toarray()
#        temp=temp.astype(np.uint64)
#        heidi_matrix=heidi_matrix*2 + temp
#        temp_array.append([heidi_matrix])
#        factor=factor*2
    
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
    
#    fig=plt.figure(2)
#    ax2 = fig.add_subplot(111,projection='3d')
#    col1 = ax2.scatter(cluster_matrix[:,0],cluster_matrix[:,1],cluster_matrix[:,2],c=labels)
#    plt.show()
#    
#    fig=plt.figure(3)
#    ax3 = fig.add_subplot(111,projection='3d')
#    col1 = ax3.scatter(cluster_matrix[:,0],cluster_matrix[:,1],cluster_matrix[:,3],c=labels)
#    plt.show()
#    
#    fig=plt.figure(4)
#    ax4 = fig.add_subplot(111,projection='3d')
#    col1 = ax4.scatter(cluster_matrix[:,0],cluster_matrix[:,1],cluster_matrix[:,4],c=labels)
#    plt.show()
    
   
   
    
    
    
    
    
    
