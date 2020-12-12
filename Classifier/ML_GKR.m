function [Outputs,Pre_Labels]=ML_GKR(train_data,train_target,test_data,test_target,para)
%ML_GKR trains a multi-label classifier based on Gaussian Kernel
%Regression, and output the results of prediction.
%
%    Syntax
%
%      [Outputs,Pre_Labels]=ML_GKR(train_data,train_target,test_data,test_target,para)
%
%    Description
%
%       ML_GKR takes,
%           train_data   - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_target - A QxM array, if the ith training instance belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_data        - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target      - A QxM2 array, if the ith testing instance belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           para       - window wide parameter; default 10
%      and returns,
%           Outputs
%           Pre_Labels 
%           
%
[x_dim,y_dim]=size(test_data);
[q,m]=size(test_target);
h=(para).*ones(y_dim,1);
 for j=1:m
      y=train_target;
      x=train_data;
    
      xs=test_data(:,j);
     for i=1:q
         Outputs(i,j)=gaussian_kern_reg(xs,train_data,train_target(i,:),h); % Prediction
     end
 end
 
for j=1:m
    for i=1:q
        if(Outputs(i,j)>=0)
            Pre_Labels(i,j)=1;
        else
            Pre_Labels(i,j)=-1;
        end
    end
end






