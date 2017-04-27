# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 08:02:29 2017

@author: Ayushi

this code is for reading dataset
"""
import pandas as pd
import numpy as np

def readIrisDataset():
    inputData=pd.read_csv(filepath_or_buffer='./dataset/iris/iris.data',sep=',',header=None)
    row=len(inputData.index)
    col=len(inputData.columns)
    feature_vector=inputData.iloc[:,0:col-1]
    classLabel_given=inputData.iloc[:,col-1]
    classLabel_numeric=classLabel_given.astype('category').cat.codes            
    col=col-1
    class_label_dict = dict(zip(classLabel_numeric, classLabel_given))
    return feature_vector, classLabel_numeric,class_label_dict, row, col

def readHabermanDataset():
    inputData=pd.read_csv(filepath_or_buffer='./dataset/haberman/haberman.data',sep=',',header=None)    
    row=len(inputData.index)
    col=len(inputData.columns)
    feature_vector=inputData.iloc[:,0:col-1]
    classLabel_given=inputData.iloc[:,col-1]
    classLabel_numeric=classLabel_given.astype('category').cat.codes
    class_label_dict = dict(zip(classLabel_numeric, classLabel_given))
    col=col-1
    return feature_vector, classLabel_numeric,class_label_dict, row, col

def readNBADataset():
    inputData=pd.read_csv(filepath_or_buffer='./dataset/NBA/player.csv',sep=',')    
    row=len(inputData.index)
    col=len(inputData.columns)
    feature_vector=inputData.iloc[:,0:col-1]
    classLabel_given=inputData.iloc[:,col-1]
    classLabel_numeric=classLabel_given.astype('category').cat.codes            
    col=col-1
    class_label_dict = dict(zip(classLabel_numeric, classLabel_given))
    return feature_vector, classLabel_numeric,class_label_dict, row, col


def readNBADataset_new():
    inputData=pd.read_csv(filepath_or_buffer='./dataset/NBA/player1_new.csv',sep=',')    
    row=len(inputData.index)
    col=len(inputData.columns)
    feature_vector=inputData.iloc[:,0:col]
    classLabel_numeric=pd.DataFrame(np.ones_like(inputData.iloc[:,1]))            
    class_label_dict = dict(zip(classLabel_numeric, classLabel_numeric))
    return feature_vector, classLabel_numeric,class_label_dict, row, col
    
def readHDI_Dataset():
    inputdata=pd.read_csv(filepath_or_buffer='./dataset/HDI/HDI-2.csv',sep=',')
    inputdata=inputdata.replace('..',np.NaN)
    inputdata=inputdata.replace('—',np.NaN)
    inputdata.drop('Note b', axis=1, inplace=True)
    inputdata.drop('Note c', axis=1, inplace=True)
    inputdata.drop('Notes a and c', axis=1, inplace=True)
    row=len(inputdata.index)
    col=len(inputdata.columns)
    
    inputdata.iloc[:,1]=inputdata.iloc[:,1].astype('category').cat.codes
    
    for i in range(0,col):
        inputdata.iloc[:,i]=inputdata.fillna(inputdata.iloc[:,i].astype(float).mean())
    
    #inputdata.to_csv('./dataset/HDI-1_pre_processed.csv',encoding='utf-8',index=False)
    
    classLabel_numeric=pd.DataFrame(np.ones_like(inputdata.iloc[:,1]))
    class_label_dict = dict(zip(classLabel_numeric, classLabel_numeric))
    return inputdata,classLabel_numeric,class_label_dict,row,col

if __name__=='__main__':
    feature_vector, classLabel_numeric,class_label_dict, row, col=readNBADataset()

