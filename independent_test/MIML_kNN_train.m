function [Weights,tr_time]=MIML_kNN_train(train_bags,train_target,num_ref,num_citer)

   
    start_time=cputime;
    
    [num_class,num_train]=size(train_target);
    
    disp('Computing distance...');
    Dist=zeros(num_train,num_train);
    for i=1:(num_train-1)
        if(mod(i,100)==0)
            disp(strcat(num2str(i),'/',num2str(num_train)));
        end
        for j=(i+1):num_train
            Dist(i,j)=GMIL_Hausdorff(train_bags{i,1},train_bags{j,1});
        end
    end
    Dist=Dist+Dist';
    
    disp('Estimating parameters...');
    A=zeros(num_train,num_class);
    B=zeros(num_train,num_class);
    
    sorted_index=cell(num_train,1);%the neighbors for the i-th bag is stored in sorted_index{i,1} in from nearest to furtherest
    for i=1:num_train
        dist_row=Dist(i,:);
        dist_row(1,i)=-1;
        [sorted_dist_row,ref_index]=sort(dist_row,'ascend');
        sorted_index{i,1}=ref_index(2:num_train);
    end
    for i=1:num_train
        if(mod(i,100)==0)
            disp(strcat(num2str(i),'/',num2str(num_train)));
        end        
        ref_index=sorted_index{i,1}(1:num_ref);
        citer_index=[];
        for j=1:num_train
            if(ismember(i,sorted_index{j,1}(1:num_citer)))
                citer_index=[citer_index,j];
            end
        end
        target=train_target(:,[ref_index,citer_index]);
        count=sum((target==1),2)';
        A(i,:)=count;        
        B(i,:)=train_target(:,i)';
    end
    
    Weights=A\B;
    
    tr_time=cputime-start_time;