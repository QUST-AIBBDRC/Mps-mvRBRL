clear all
clc
%导入数据
%Q是真实数据,W是得分
 load('Data-DC.mat')
load('Pgoxinxi.mat')
 Q=nelabel;
 
predict_label=P;
decision_values=W;
% %处理数据
%  Q(Q==-1)=0;
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
decision_values=XX;
% tpr=[];
% fpr=[];
% thresholds=[];
% 
% for i=1:size(W,2)
% %[tpr(:,i),fpr(:,i),thresholds] =roc(Q',XX(:,i)');
% [fpr(:,i),tpr(:,i),t,auc_1(i)] =perfcurve(Q(:,i),XX(:,i),'1');
% end
% FPR=mean(fpr,2);
% TPR=mean(tpr,2);
% % %计算auc
% auc= sum((FPR(2:length(FPR),1)-FPR(1:length(FPR)-1,1)).*TPR(2:length(TPR),1))
% plot(FPR,TPR),xlabel('FPR'),ylabel('TPR');%输出ROC曲线
% auc_1;
label_or_decision='decision'; % use label('label') as decision or decision_values('decision') as decision decision will be better.
PRC_or_ROC=0;    
for i=1:size(W,2)

%     c_start=sum(num_in_class(1:ci-1))+1;%开始位置
%     c_end=sum(num_in_class(1:ci)); %结束位置
%     
%     targs=-ones(1,length(actual_label));
%     positive_i=find(actual_label==ci);%在真实标签里面，找出每类的位置
%     targs(positive_i)=1;      %将这些位置赋值为1，重新得到真实样本的标签标示，用（1，-1）
    
    targs=Q(:,i);
    
        dec=decision_values(:,i);  %第ci类的概率值
        pre_nag=find(predict_label(:,i)~=1);    %预测标签不等于ci类的位置
        dec(pre_nag)=-dec(pre_nag);  %对预测错误的位置，得分取负
        dvs=dec;

    
    [TPR_emp(:,i), FPR_emp(:,i), PPV_emp(:,i)]=draw_prc(targs,dvs,PRC_or_ROC)
end
  TPR_emp(:,2)=[];
  PPV_emp(:,2)=[]; 

TPR=mean(TPR_emp,2);
 PPV=mean( PPV_emp,2)

PPV(520)=1; 
 AP=sum(abs((TPR(2:length(TPR),1)-TPR(1:length(TPR)-1,1))).*PPV(2:length(PPV),1));
  


   save P-DC-PRC.mat TPR PPV AP
% figure;
%  plot(TPR_emp1, PPV_emp1,'*');
%     axis([0 1 0 1]);
%    title('PR curves');
%     xlabel('Recall'); ylabel('Precision');
%     set(gca, 'box', 'on');