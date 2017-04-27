# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:37:07 2017

@author: Ayushi
"""

from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import orderPoints as op
import readDataset as rd
import matplotlib.pyplot as plt

def generateHeidiMatrix(sorted_data,row,col,nofKNN=10):
    k=nofKNN #1-NN
    heidi_matrix=np.zeros(shape=(row,row),dtype=np.uint64)
    max_count=int(math.pow(2,col))
    allsubspaces=range(1,max_count)
    f=lambda a:sorted(a,key=lambda x:sum(int(d)for d in bin(x)[2:]))
    allsubspaces=f(allsubspaces)
    frmt=str(col)+'b'
    factor=1
    bit_subspace={}
    count=0
    for i in allsubspaces:
        bin_value=str(format(i,frmt))
        bin_value=bin_value[::-1]
        subspace_col=[index for index,value in enumerate(bin_value) if value=='1']
        print ("%d : %s : '%s'" %(i,subspace_col,bin_value[::-1]))
        bit_subspace[count]=subspace_col
        count=count+1
        subspace=sorted_data.iloc[:,subspace_col]    
        np_subspace=subspace.values
        nbrs=NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(np_subspace)
        temp=nbrs.kneighbors_graph(np_subspace).toarray()
        temp=temp.astype(np.uint64)
        heidi_matrix=heidi_matrix + temp*factor
        factor=factor*2    
    return heidi_matrix,bit_subspace

def visualizeHeidiImage(heidi_matrix,bit_subspace):
    max_count=len(bit_subspace)  
    r=int(max_count/3)
    g=int(max_count/3)
    b=max_count-r-g
    
    x=heidi_matrix>>(max_count-r)
    y=(heidi_matrix & ((pow(2,g)-1)<<b)) >> b
    z=(heidi_matrix & (pow(2,b)-1))

    heidi_img=np.dstack((x,y,z))    
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.imshow(heidi_img, interpolation='nearest',picker=True)
    plt.savefig('nba_new.png')
    plt.show()
    cluster_matrix=[]
    for i in range(0,row):
        for j in range(0,row):
            cluster_matrix.append([i,j,x[i][j],y[i][j],z[i][j]]) 
            
    cluster_matrix=np.array(cluster_matrix)
    #cluster_matrix=cluster_matrix.T
    return cluster_matrix,heidi_img
        
if __name__=="__main__":
    #feature_vector, classLabel_given,class_label_dict, row, col=rd.readHabermanDataset() 
    feature_vector, classLabel_given,class_label_dict, row, col=rd.readNBADataset_new()
    feature_vector.loc[:,'classLabel']=classLabel_given.iloc[:,0]    
    sorted_data=op.sortbasedOnclassLabel(feature_vector,'centroid_distance')
    heidi_matrix,bit_subspace=generateHeidiMatrix(sorted_data,row,col,nofKNN=10)
    np.savetxt("nba_new.csv", heidi_matrix, delimiter=",")
    cluster_matrix,heidi_img=visualizeHeidiImage(heidi_matrix,bit_subspace)
    
