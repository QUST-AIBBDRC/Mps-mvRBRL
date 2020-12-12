
import scipy.io as sio
import numpy as np 
import pickle as p 
from sklearn.preprocessing import scale,StandardScaler,MinMaxScaler 
import matplotlib.pyplot as plt

np.random.seed(100)
first = sio.loadmat('tpc1_python_PSSM.mat')
yeast_PA=first.get('pssm')
#yeast_PB=first.get('yeast_PB')
index_PA=first.get('index_PA')
#index_PB=first.get('index_PB')
CNN_pre_A=[]
#CNN_pre_B=[]
num=len(index_PA)

#row1=4
data_list1=[]
#data_list2=[]
#data_list3=[]
#data_list4=[]
data_H=[]
for i in range(num-1):
    H1=index_PA[i].tolist()[0]###tolist将数组或者矩阵转换成列表
    H2=index_PA[i+1].tolist()[0]
#    H2=index_PA[i+1].tolist()[0]
    #data_H=yeast_PA[0:H1,:]
    data_H=yeast_PA[H1:H2,:]####data_H表示每条序列对应的pssm矩阵
    #data_M=cv2.resize(data_H, (20,20), interpolation=cv2.INTER_AREA) 
    data_list1.append(data_H)
    H1=[]
    H2=[]
    data_H=[]
#######################################
#for i in range(num-1):
#    H1=index_PB[i].tolist()[0]
#    H2=index_PB[i+1].tolist()[0]
#    data_H=yeast_PB[H1:H2]
#    data_list2.append(data_H)
#    H1=[]
#    H2=[]
#    data_H=[]
####################################### 
f1 = open('3681_pssm—TPC.data', 'wb') ###建立一个名为676_pssm_PA1.data的空的.data文件
p.dump(data_list1, f1) ##把data_list1放进上边建立的空.data文件
f1.close() 
#sio.savemat('tpc1_676_pssm_PB.data',{'data_list1':data_list1})
#f2 = open('Mmusc_pssm_PB.data', 'wb') 
#p.dump(data_list2, f2) 
#f2.close()     
