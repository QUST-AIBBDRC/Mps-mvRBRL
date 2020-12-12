# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:24:15 2020

@author: Administrator
"""

import numpy as np
import scipy.io as sio
import pickle as p
import time
import pickle as p
import numpy as np
import pandas as pd
#import xgboost as xgb
#import lightgbm as lgb
import scipy.io as sio
from sklearn import svm
from sklearn.svm import SVC
#import utils.tools as utils

def average(matrixSum, seqLen):
    # average the summary of rows
    matrix_array = np.array(matrixSum)
    matrix_array = np.divide(matrix_array, seqLen)
    matrix_array_shp = np.shape(matrix_array)
    matrix_average = [(np.reshape(matrix_array, (matrix_array_shp[0] * matrix_array_shp[1], )))]
    return matrix_average

def preHandleColumns(PSSM,STEP,ID):
    PSSM=PSSM.astype(float)
    matrix_final = [ [0.0] * 20 ] * 20
    matrix_final=np.array(matrix_final)
    seq_cn=np.shape(PSSM)[0]
    if ID==0:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j]+=(PSSM[k][i]*PSSM[k+STEP][j])

    elif ID==1:
        for i in range(20):
            for j in range(20):
                for k in range(seq_cn - STEP):
                    matrix_final[i][j] += ((PSSM[k][i]-PSSM[k+STEP][j]) * (PSSM[k][i]-PSSM[k+STEP][j])/4.0)
    return matrix_final

def tpc(input_matrix):
    #print "start tpc function"
    #PART=0
    STEP=1
    ID=0
    #KEY=1
    #matrix_final=preHandleColumns(input_matrix, STEP, PART, ID)
    matrix_final=preHandleColumns(input_matrix, STEP, ID)
    matrix_tmp=[0.0] * 20
    matrix_tmp=np.array(matrix_tmp)
    for i in range(20):
        matrix_tmp=list(map(sum,zip(matrix_final[i], matrix_tmp)))
        #map函数它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
    for i in range(20):
        for j in range(20):
            matrix_final[i][j]=matrix_final[i][j]/matrix_tmp[j]
    tpc_vector = average(matrix_final, 1.0)
    #print "end tpc function"
    return tpc_vector

#f1=open(r'tpc1_66_pssm_PB.data.mat','rb')
f1=open(r'3681_pssm—TPC.data','rb')
pssm1=p.load(f1)
aac=[]
dpc=[]
#for i in range(len(pssm1)):
#    aac_pssm_obtain=aac_pssm(pssm1[i])
#    aac.append(aac_pssm_obtain)
for i in range(len(pssm1)):
    dpc_pssm_obtain=tpc(pssm1[i])
    
    dpc.append(dpc_pssm_obtain)

dpc1=np.array(dpc)
[m,n,q]=np.shape(dpc1)
X1=np.reshape(dpc1,(m,400))
sio.savemat('PSSM_TPC.mat',{'3681_pssm':X1})



#f1 = open('676_pssm_dpc.data', 'wb') 
#p.dump(dpc1, f1) 
#f1.close() 
#sio.savemat('Hsapi_aac_pssm_PB.mat',{'aac_pssm_PB':aac})

