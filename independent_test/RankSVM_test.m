function [Outputs,Pre_Labels]=RankSVM_test(test_data,test_target,svm,Weights,Bias,SVs,Weights_sizepre,Bias_sizepre)

[num_testing,tempvalue]=size(test_data);
[num_class,tempvalue]=size(test_target);
[tempvalue,num_training]=size(SVs);

Label=cell(num_testing,1);
not_Label=cell(num_testing,1);
Label_size=zeros(1,num_testing);
size_alpha=zeros(1,num_testing);
for i=1:num_testing
    temp=test_target(:,i);
    Label_size(1,i)=sum(temp==ones(num_class,1));
    size_alpha(1,i)=Label_size(1,i)*(num_class-Label_size(1,i));
    for j=1:num_class
        if(temp(j)==1)
            Label{i,1}=[Label{i,1},j];
        else
            not_Label{i,1}=[not_Label{i,1},j];
        end
    end
end

kernel=zeros(num_testing,num_training);
if(strcmp(svm.type,'RBF'))
    for i=1:num_testing
        for j=1:num_training
            gamma=svm.para;
            kernel(i,j)=exp(-gamma*sum((test_data(i,:)'-SVs(:,j)).^2))
        end
    end
else
    if(strcmp(svm.type,'Poly'))
        for i=1:num_testing
            for j=1:num_training
                gamma=svm.para(1);
                coefficient=svm.para(2);
                degree=svm.para(3);
                kernel(i,j)=(gamma*test_data(i,:)*SVs(:,j)+coefficient)^degree;
            end
        end
    else
        for i=1:num_testing
            for j=1:num_training
                kernel(i,j)=test_data(i,:)*SVs(:,j);
            end
        end
    end
end

Outputs=zeros(num_class,num_testing);
for i=1:num_testing
    for k=1:num_class
        temp=0;
        for j=1:num_training
            temp=temp+Weights(k,j)*kernel(i,j);            
        end
        temp=temp+Bias(k);
        Outputs(k,i)=temp;
    end
end
Threshold=([Outputs',ones(num_testing,1)]*[Weights_sizepre,Bias_sizepre]')';
Pre_Labels=zeros(num_class,num_testing);
for i=1:num_testing
    for k=1:num_class
        if(Outputs(k,i)>=Threshold(1,i))
            Pre_Labels(k,i)=1;
        else
            Pre_Labels(k,i)=-1;
        end
    end
end

