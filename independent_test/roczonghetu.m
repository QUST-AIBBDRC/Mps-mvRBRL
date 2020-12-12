clear all
clc
%导入数据
 load('Data-rong.mat')
load('Pgoxinxi.mat')
%Q是真实数据,W是得分
%  Q=xlsread('2.xlsx','真实标签');
% W=xlsread('2.xlsx','预测得分');
Q=nelabel;
%   Q(Q==-1)=0;
%处理数据

% %输出概率矩阵要求输出概率在0-1。变换标签得分
XX=[];
for j=1:size(W,2)
    %把小于0的转换到0-0.5,计算斜率和截距
    x1(j)=min(W(:,j));
    x2=0;
    y1=0;
    y2=0.5;
    k1=(y2-y1)/(x2-x1(j));
    b1=0.5;
    %把大于0的转换到0.5-1
    x3=0;
    x4(j)=max(W(:,j));
    y3=0.5;
    y4=1;
    k2=(y4-y3)/(x4(j)-x3);
    b2=0.5;
  %重新计算各个元素的得分
for i=1:size(W,1)
    if W(i,j)>=0
      XX(i,j)=k2*W(i,j)+b2;
    end
     if W(i,j)<0
        XX(i,j)=k1*W(i,j)+b1;
    end
    
end
end
tpr=[];
fpr=[];
thresholds=[];


  for i=1:size(W,2)
  [fpr(:,i),tpr(:,i),t,auc_1(i)] =perfcurve(Q(:,i),XX(:,i),'1');
  end
%      tpr(:,6)=[];tpr(:,7)=[];
%    fpr(:,6)=[];fpr(:,7)=[];

 FPR=mean(fpr,2);
 TPR=mean(tpr,2);
% % %计算auc
  auc= sum((FPR(2:length(FPR),1)-FPR(1:length(FPR)-1,1)).*TPR(2:length(TPR),1))
%   plot(FPR,TPR),xlabel('FPR'),ylabel('TPR');%输出ROC曲线
% auc_1; 
% 
     save P-rong-ROC.mat FPR TPR auc