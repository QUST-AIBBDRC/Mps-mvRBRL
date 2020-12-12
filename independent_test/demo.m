transform_type=5; %降维方法
transform_parameter.ratio =50; %降的维数
%% 融合
load('goxinxi.mat')
X=gozhengli;
L=nelabel;
[YANGBEN,WEISHU]=size(L);
% shu=zscore(shushu);
%数据处理
train_data = X;
train_label = L;
[N, D] = size(train_data);
mean_train = mean(train_data);
train_CX = train_data - repmat(mean_train,N,1); %centralizing training features
mean_label = mean(train_label);
train_CY = train_label - repmat(mean_label, N, 1);
currentTrainData = train_CX;
%降维方法
transform_parameter.rank =1;
switch(transform_type)
    case 1 %PCA

        
    case 2 %MLSI
        transform_parameter.beta = 0.5;
        transform_parameter.regXY = 0.1;
        transform_parameter.regX  = 0.1;
        
        
    case 3 %MDDM
        transform_parameter.beta = 0.5;

        
    case 4 %MVMD
        transform_parameter.beta  = 0.5;
        
    otherwise % wMLDA
        %KBS
        transform_parameter.regXY = 0.01;
        
    
end
%降维操作
if (transform_type >0)
    if(transform_type >=1 && transform_type <=4) 
            [PPP] = execute_transform(currentTrainData, train_CY, transform_type, transform_parameter); 
    else 
            [PPP] = execute_transform(currentTrainData, train_label, transform_type, transform_parameter);
    end
    
    current = currentTrainData * PPP;        
end

%%自行添加各分类器所需的参数
P=[];W=[];
NIter =transform_parameter.ratio;

lambda1 = 1;
lambda2 = 0.01;
lambda3 =0.1;

for i=1:YANGBEN
    C=current;
    B=L;
    test_data=C(i,:);test_target=B(i,:);
    C(i,:)=[]; B(i,:)=[];
    train_data=C;train_target=B;

num_feature_origin = size(train_data, 2);
train_data(:, num_feature_origin + 1) = 1;
test_data(:, num_feature_origin + 1) = 1;

[ A, obj ] = train_linear_RBRL_APG( train_data, train_target, lambda1, lambda2, lambda3, NIter );
[ pre_Label, pre_F ] = Predict( test_data, A );

% [pre_F,pre_Label]=ML_GKR(train_data,train_target,test_data,test_target,para);
% [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data,train_target,Num,Smooth);
%  [pre_F,pre_Label]=MLKNN_test(train_data,train_target,test_data,test_target,Num,Prior,PriorN,Cond,CondN);
% [pre_F,pre_Label]=LIFT(train_data,train_target,test_data,test_target,ratio,svm);

 P=[P;pre_Label];
 W=[W;pre_F];
clear A B
end
%% 评价指标
HL=Hamming_loss(P,L);
AP=Average_precision(W,L);
CV=coverage_new(P,L);
RL=Ranking_loss(W,L);


OAA=0;
for i=1:YANGBEN
if P(i,:)==L(i,:);
OAA=OAA+1;
end
end
 zhengque(:,:)=gelei(L,P,WEISHU,YANGBEN);