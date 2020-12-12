clear all
clc
load shuju
pssm=[];
maxlen=[];
for i=1:numel(C)
    data=C{i};
    data= 1.0 ./ ( 1.0 + exp(-data) );
    pssm=[pssm;data];  %%%%把所有的PSSM文件前L行20列保存到一个文件里
    [row,column]=size(data);
    maxlen=[maxlen;row];
    data=[];
    row=[];
end
index_PA=cumsum(maxlen);   %%%cumsum函数通常用于计算一个数组各行的累加值，index_PA得到所有的行数
maxlen=[];
index_PA=[0;index_PA];
[m,n]=size(index_PA);
save tpc1_python_PSSM.mat pssm index_PA 
