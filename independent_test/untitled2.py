# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 09:11:33 2020

@author: del
"""

import itertools
import turtle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib as mpl
import matplotlib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import pandas as pd

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
lw=1#线的粗细
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(10,5))


plt.subplot(121)
##############

data_train = sio.loadmat(r'E:\ROCAPR\P-PAAC-ROC.mat')
fpr=data_train.get('FPR')
tpr=data_train.get('TPR')
auc_score_class1=data_train.get('auc')  
plt.plot(fpr, tpr, color='orange',
lw=lw, label='PseAAC (AUC = %.4f)' % auc_score_class1)
#
#
data_train = sio.loadmat(r'E:\ROCAPR\P-PseSSM-ROC.mat')
fpr=data_train.get('FPR')
tpr=data_train.get('TPR')
auc_score_class2=data_train.get('auc')  
plt.plot(fpr, tpr, color='darkorchid',
lw=lw, label='PsePSSM (AUC = %.4f)' % auc_score_class2)


data_train = sio.loadmat(r'E:\ROCAPR\P-DC-ROC.mat')
fpr=data_train.get('FPR')
tpr=data_train.get('TPR')
auc_score_class3=data_train.get('auc')  
plt.plot(fpr, tpr, color='forestgreen',
lw=lw, label='DC (AUC = %.4f)' % auc_score_class3)


data_train = sio.loadmat(r'E:\ROCAPR\P-PSSMTPC-ROC.mat')
fpr=data_train.get('FPR')
tpr=data_train.get('TPR')
auc_score_class4=data_train.get('auc')  
plt.plot(fpr, tpr, color='deepskyblue',
lw=lw, label='PSSM-TPC (AUC = %.4f)' % auc_score_class4)

data_train = sio.loadmat(r'E:\ROCAPR\P-GO-ROC.mat')
fpr=data_train.get('FPR')
tpr=data_train.get('TPR')
auc_score_class5=data_train.get('auc')  
plt.plot(fpr, tpr, color='#7FFF00',
lw=lw, label='GO (AUC = %.4f)' % auc_score_class5)

data_train = sio.loadmat(r'E:\ROCAPR\P-rong-ROC.mat')
fpr=data_train.get('FPR')
tpr=data_train.get('TPR')
auc_score_class6=data_train.get('auc')  
plt.plot(fpr, tpr, color='black',
lw=lw, label='ALL (AUC = %.4f)' % auc_score_class6)

data_train = sio.loadmat(r'E:\ROCAPR\P-DErong-ROC.mat')
fpr=data_train.get('FPR')
tpr=data_train.get('TPR')
auc_score_class7=data_train.get('auc')  
plt.plot(fpr, tpr, color='#FF0000',
lw=lw, label='ALL(DE) (AUC = %.4f)' % auc_score_class7)

ax = plt.gca()
plt.tick_params(labelsize=10) 
labels = ax.get_xticklabels() + ax.get_yticklabels()
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
font = {'family': 'Times New Roman', 'color': 'black', 'size': 10}
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.03, 1.01])
plt.ylim([0, 1.03])
plt.xlabel('False positive rate',font)
plt.ylabel('True positive rate',font)
font_L = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
legend = plt.legend(prop=font_L,loc="lower right")
#
#
plt.subplot(122)
###############

#
data_trainP = sio.loadmat(r'E:\ROCAPR\P-PAAC1-PRC.mat')
fpr=data_trainP.get('TPR')#取出字典里的data
tpr=data_trainP.get('PPV')
aupr1=data_trainP.get('AP')
plt.plot(fpr, tpr,  color='orange',
lw=lw, label='PseAAC (AUPR=%.4f)' % aupr1)

data_trainP = sio.loadmat(r'E:\ROCAPR\P-PseSSM1-PRC.mat')
fpr=data_trainP.get('TPR')#取出字典里的data
tpr=data_trainP.get('PPV')
aupr2=data_trainP.get('AP')
plt.plot(fpr, tpr,  color='darkorchid',
lw=lw, label='PsePSSM (AUPR=%.4f)' % aupr2)

data_trainP = sio.loadmat(r'E:\ROCAPR\P-PSSMTPC1-PRC.mat')
fpr=data_trainP.get('TPR')#取出字典里的data
tpr=data_trainP.get('PPV')
aupr3=data_trainP.get('AP')
plt.plot(fpr, tpr,  color='deepskyblue',
lw=lw, label='PSSM-TPC (AUPR=%.4f)' % aupr3)

data_trainP = sio.loadmat(r'E:\ROCAPR\P-DC1-PRC.mat')
fpr=data_trainP.get('TPR')#取出字典里的data
tpr=data_trainP.get('PPV')
aupr4=data_trainP.get('AP')
plt.plot(fpr, tpr,  color='forestgreen',
lw=lw, label='DC (AUPR=%.4f)' % aupr4)

data_trainP = sio.loadmat(r'E:\ROCAPR\P-GO1-PRC.mat')
fpr=data_trainP.get('TPR')#取出字典里的data
tpr=data_trainP.get('PPV')
aupr5=data_trainP.get('AP')
plt.plot(fpr, tpr,  color='#7FFF00',
lw=lw, label='GO (AUPR=%.4f)' % aupr5)

data_trainP = sio.loadmat(r'E:\ROCAPR\P-rong-PRC.mat')
fpr=data_trainP.get('TPR')#取出字典里的data
tpr=data_trainP.get('PPV')
aupr6=data_trainP.get('AP')
plt.plot(fpr, tpr,  color='black',
lw=lw, label='ALL (AUPR=%.4f)' % aupr6)

data_trainP = sio.loadmat(r'E:\ROCAPR\P-DErong-PRC.mat')
fpr=data_trainP.get('TPR')#取出字典里的data
tpr=data_trainP.get('PPV')
aupr7=data_trainP.get('AP')
plt.plot(fpr, tpr,  color='#FF0000',
lw=lw, label='ALL(DE) (AUPR=%.4f)' % aupr7)


aupr= []
aupr.append([aupr1,aupr2,aupr3,aupr4,aupr5,aupr6,aupr7])
auprM=np.array(aupr)
ax = plt.gca()
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
font = {'family': 'Times New Roman', 'color': 'black', 'size': 10}
ax.set_ylabel('Recall',fontdict=font)     
ax.set_xlabel('Precision',fontdict=font)
plt.xlim([0,1.01])
plt.ylim([0.31,1.02])
plt.xlabel('Recall',fontsize=10)
plt.ylabel('Precision',fontsize=10)
font_L = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
legend = plt.legend(prop=font_L,loc="lower left")
#
#aupr_SUM= []
#aupr_SUM=np.vstack([auprM])
#aupr_SUM.append([auprA;auprB])
#aupr_sum=np.array(aupr_SUM)
#data_csv = pd.DataFrame(data=aupr_SUM)
#data_csv.to_csv('aupr_sum.csv')
plt.savefig('5tezheng_ROC_PR2.tif',dpi=300,format='tif')
plt.show()





