clear all
clc
load shuju
pssm=[];
maxlen=[];
for i=1:numel(C)
    data=C{i};
    data= 1.0 ./ ( 1.0 + exp(-data) );
    pssm=[pssm;data];  %%%%�����е�PSSM�ļ�ǰL��20�б��浽һ���ļ���
    [row,column]=size(data);
    maxlen=[maxlen;row];
    data=[];
    row=[];
end
index_PA=cumsum(maxlen);   %%%cumsum����ͨ�����ڼ���һ��������е��ۼ�ֵ��index_PA�õ����е�����
maxlen=[];
index_PA=[0;index_PA];
[m,n]=size(index_PA);
save tpc1_python_PSSM.mat pssm index_PA 
