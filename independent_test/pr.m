clear all
clc
%��������
%Q����ʵ����,W�ǵ÷�
 load('Data-DC.mat')
load('Pgoxinxi.mat')
 Q=nelabel;
 
predict_label=P;
decision_values=W;
% %��������
%  Q(Q==-1)=0;
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
% % %����auc
% auc= sum((FPR(2:length(FPR),1)-FPR(1:length(FPR)-1,1)).*TPR(2:length(TPR),1))
% plot(FPR,TPR),xlabel('FPR'),ylabel('TPR');%���ROC����
% auc_1;
label_or_decision='decision'; % use label('label') as decision or decision_values('decision') as decision decision will be better.
PRC_or_ROC=0;    
for i=1:size(W,2)

%     c_start=sum(num_in_class(1:ci-1))+1;%��ʼλ��
%     c_end=sum(num_in_class(1:ci)); %����λ��
%     
%     targs=-ones(1,length(actual_label));
%     positive_i=find(actual_label==ci);%����ʵ��ǩ���棬�ҳ�ÿ���λ��
%     targs(positive_i)=1;      %����Щλ�ø�ֵΪ1�����µõ���ʵ�����ı�ǩ��ʾ���ã�1��-1��
    
    targs=Q(:,i);
    
        dec=decision_values(:,i);  %��ci��ĸ���ֵ
        pre_nag=find(predict_label(:,i)~=1);    %Ԥ���ǩ������ci���λ��
        dec(pre_nag)=-dec(pre_nag);  %��Ԥ������λ�ã��÷�ȡ��
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