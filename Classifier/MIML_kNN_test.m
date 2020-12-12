function [Outputs,Pre_Labels]=MIML_kNN_test(train_bags,train_target,test_bags,test_target,num_ref,num_citer,Weights)
%MIML_kNN_test tests a lazy multi-instance multi-label learner.
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,te_time]=MIML_kNN_test(train_bags,train_target,test_bags,test_target,num_ref,num_citer,Weights)
%
%    Description
%
%       MIML_kNN_test takes,
%           train_bags       - An M1x1 cell, the ith training bag is stored in train_bags{i,1}
%           train_target     - A QxM1 array, if the ith training bag belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           test_bags        - An M2x1 cell, the ith test bag is stored in test_bags{i,1}
%           test_target      - A QxM2 array, if the ith test bag belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           num_ref          - Number of references considered by LML
%           num_citer        - Number of citers considered by LML
%           Weights          - A QxQ matrix used for label prediction
%      and returns,
%           HammingLoss      - The hamming loss on testing data
%           RankingLoss      - The ranking loss on testing data
%           OneError         - The one-error on testing data as
%           Coverage         - The coverage on testing data as
%           Average_Precision- The average precision on testing data
%           Outputs          - A QxM2 array, the probability of the ith testing instance belonging to the jCth class is stored in Outputs(j,i)
%           Pre_Labels       - A QxM2 array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%           te_time          - The time spent in testing

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
 