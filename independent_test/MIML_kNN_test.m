function [Outputs,Pre_Labels]=MIML_kNN_test(train_bags,train_target,test_bags,test_target,num_ref,num_citer,Weights)

    start_time=cputime;
    
    [num_class,num_train]=size(train_target);
    num_test=size(test_bags,1);
    num_inst=num_train+num_test;
    bags=[train_bags;test_bags];
    
    disp('Computing distance...');
    Dist=zeros(num_inst,num_inst);
    for i=1:(num_inst-1)
        if(mod(i,100)==0)
            disp(strcat(num2str(i),'/',num2str(num_inst)));
        end
        for j=(i+1):num_inst
            Dist(i,j)=GMIL_Hausdorff(bags{i,1},bags{j,1});
        end
    end
    Dist=Dist+Dist';
    
    sorted_index=cell(num_inst,1);%the neighbors for the i-th bag is stored in sorted_index{i,1} in from nearest to furtherest
    for i=1:num_inst
        dist_row=Dist(i,:);
        dist_row(1,i)=-1;
        [sorted_dist_row,ref_index]=sort(dist_row,'ascend');
        sorted_index{i,1}=ref_index(2:num_inst);
    end
    
    A=zeros(num_test,num_class);
    
    for i=1:num_test
        if(mod(i,10)==0)
            disp(strcat(num2str(i),'/',num2str(num_test)));
        end
        temp_index=sorted_index{(i+num_train),1};
        filter_index=find(temp_index>num_train);
        temp_index(filter_index)=[];
        ref_index=temp_index(1:num_ref);
        citer_index=[];
        for j=1:num_train
            temp_index=sorted_index{j,1};
            tmp=find(temp_index==i+num_train);
            temp_index(1,tmp)=j;
            filter_index=find(temp_index>num_train);
            temp_index(filter_index)=[];
            if(ismember(j,temp_index(1:num_citer)))
                citer_index=[citer_index,j];
            end
        end
        target=train_target(:,[ref_index,citer_index]);
        count=sum((target==1),2)';
        A(i,:)=count;
    end
    
    Outputs=(A*Weights)';
    
%Evaluation
    Pre_Labels=zeros(num_class,num_test);
    for i=1:num_test
        for j=1:num_class
            if(Outputs(j,i)>=0)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=-1;
            end
        end
    end
  