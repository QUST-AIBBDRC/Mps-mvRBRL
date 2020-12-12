function [Outputs,Pre_Labels]=ML_GKR(train_data,train_target,test_data,test_target,para)


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






