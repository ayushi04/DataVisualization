# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 08:35:24 2017

@author: Ayushi
"""
import pandas as pd
import readDataset as rd
from scipy.spatial import distance
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


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
 

def sortbasedOnclassLabel(feature_vector,ordermeasure):
    centroids=pd.DataFrame()
    for k in set(feature_vector.classLabel):
        x=feature_vector[feature_vector.classLabel==k].mean()
        centroids=centroids.append(x.to_frame().T,ignore_index=True)
    
    sorted_data=pd.DataFrame()    
    for i in set(feature_vector.classLabel):
        if(ordermeasure=='centroid_distance'):
            temp=getDistances(feature_vector[feature_vector.classLabel==i].copy(),centroids[centroids.classLabel==i].iloc[:,0:-1])
            sorted_data=pd.concat([sorted_data,temp])
            
        elif(ordermeasure=='connected_distance'): 
            temp=getConnectedDistances(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
            sorted_data=pd.concat([sorted_data,temp])
            
        elif(ordermeasure=='mst_distance'):
            temp=minSpanningTree(feature_vector[feature_vector.classLabel==i].copy(),centroids[i])
            sorted_data=pd.concat([sorted_data,temp])
    
    return sorted_data.iloc[:,0:-1]  

if __name__=="__main__":
    feature_vector, classLabel_given,class_label_dict, row, col=rd.readIrisDataset()    
    
    feature_vector.loc[:,'classLabel']=classLabel_given    
    sorted_data=sortbasedOnclassLabel(feature_vector,'centroid_distance')
            

    