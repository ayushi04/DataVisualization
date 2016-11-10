# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 09:17:23 2016
@author: Ayushi
"""

import pandas as pd
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors

k=10

data=pd.read_csv(filepath_or_buffer='C:/Users/Ayushi/Desktop/1 sem/data visualisation/data_set/haberman.data',header=None,sep=',')
col=len(data.columns)
row=len(data.index)
#for k in [5,10,15,20,25,30]:
heidi_matrix=np.zeros(shape=(row,row),dtype=np.uint64)
max_count=int(math.pow(2,col))
frmt=str(col)+'b'
for i in range (1,max_count):
    bin_value=str(format(i,frmt))
    subspace_col=[index for index,value in enumerate(bin_value) if value=='1']
    subspace=data.iloc[:,subspace_col]    
    np_subspace=subspace.as_matrix()   
    nbrs=NearestNeighbors(n_neighbors=k,algorithm='ball_tree').fit(np_subspace)
    temp=nbrs.kneighbors_graph(np_subspace).toarray()
    temp=temp.astype(np.uint64)
    heidi_matrix=(heidi_matrix*2) + temp

from matplotlib import pyplot as plt
plt.imshow(heidi_matrix, interpolation='nearest')
plt.show()

