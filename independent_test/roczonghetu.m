clear all
clc
%��������
 load('Data-rong.mat')
load('Pgoxinxi.mat')
%Q����ʵ����,W�ǵ÷�
%  Q=xlsread('2.xlsx','��ʵ��ǩ');
% W=xlsread('2.xlsx','Ԥ��÷�');
Q=nelabel;
%   Q(Q==-1)=0;
%��������

% %������ʾ���Ҫ�����������0-1���任��ǩ�÷�
XX=[];
for j=1:size(W,2)
    %��С��0��ת����0-0.5,����б�ʺͽؾ�
    x1(j)=min(W(:,j));
    x2=0;
    y1=0;
    y2=0.5;
    k1=(y2-y1)/(x2-x1(j));
    b1=0.5;
    %�Ѵ���0��ת����0.5-1
    x3=0;
    x4(j)=max(W(:,j));
    y3=0.5;
    y4=1;
    k2=(y4-y3)/(x4(j)-x3);
    b2=0.5;
  %���¼������Ԫ�صĵ÷�
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
% % %����auc
  auc= sum((FPR(2:length(FPR),1)-FPR(1:length(FPR)-1,1)).*TPR(2:length(TPR),1))
%   plot(FPR,TPR),xlabel('FPR'),ylabel('TPR');%���ROC����
% auc_1; 
% 
     save P-rong-ROC.mat FPR TPR auc