function [Centroids,Sigma_value,Weights,tr_time]=MIML_RBF_train(train_bags,train_target,ratio,mu)
%MIML_RBF_train trains a multi-instance multi-label RBF learner
%
%    Syntax
%
%       [Centroids,Sigma_value,Weights,tr_time]=MIML_RBF_train(train_bags,train_target,ratio,mu)
%
%    Description
%
%       MIML_RBF_train takes,
%           train_bags    - An Mx1 cell, the ith training bag is stored in train_bags{i,1}
%           train_target  - A QxM array, if the ith training bag belongs to the jth class, then train_target(j,i) equals +1, otherwise train_target(j,i) equals -1
%           ratio         - The number of centroids of the i-th class is set to be ratio*Ti, where Ti is the number of train bags with lable i
%           mu            - The ratio used to determine the standard deviation of the Gaussian activation function
%      and returns,
%           Centroids    - A Kx1 cell structure, where the k-th centroid of the RBF neural network is stored in Centroid{k,1}
%           Sigma_value  - A 1xK vector, where the sigma value for the k-th centroid is stored in Sigma_value(1,k)
%           Weights      - A (K+1)xQ matrix used for label prediction
%           tr_time      - The time spent in training
   
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
    
    disp('First layer clustering...');
    num_cluster=zeros(1,num_class);
    for j=1:num_class
        num_cluster(1,j)=ceil(ratio*sum(train_target(j,:)==1));
    end
    num_centroid=sum(num_cluster);
    Centroids=cell(num_centroid,1);
    counter=0;
    centroid_index=[];
    for j=1:num_class
        disp(strcat(num2str(j),'/',num2str(num_class)));
        temp_index=find(train_target(j,:)==1);
        distance_matrix=Dist(temp_index,temp_index);
        [clustering,matrix_fai,num_iter]=MIML_cluster(num_cluster(j),distance_matrix);
        for k=1:num_cluster(j)
            counter=counter+1;
            centroid_index=[centroid_index,temp_index(clustering{k,1})];
            Centroids{counter,1}=train_bags{temp_index(clustering{k,1}),1};
        end
    end
    
    centroid_dist=Dist(centroid_index,centroid_index);
    numerator=sum(sum(triu(centroid_dist,1)));
    denominator=num_centroid*(num_centroid-1)/2;
    sigma=mu*(numerator/denominator);
    
    Sigma_value=zeros(1,num_centroid);
    counter=0;
    for j=1:num_class
        sigma_j=sigma;

        for k=1:num_cluster(j)
            counter=counter+1;
            Sigma_value(1,counter)=sigma_j;
        end
    end
    
    disp('Second layer optimization...');
    A=zeros(num_train,num_centroid+1);
    B=zeros(num_train,num_class);
    
    for i=1:num_train
        if(mod(i,100)==0)
            disp(strcat(num2str(i),'/',num2str(num_train)));
        end
        
        temp_vec=zeros(1,num_centroid);
        counter=0;
        for j=1:num_class
            for k=1:num_cluster(j)
                counter=counter+1;
                temp_sigma=Sigma_value(1,counter);
                temp_vec(1,counter)=exp(-Dist(i,centroid_index(counter))^2/(2*temp_sigma^2));
            end
        end
        temp_vec=[1,temp_vec];
        A(i,:)=temp_vec;
                
        B(i,:)=train_target(:,i)';
    end
    
    Weights=A\B;
    
    tr_time=cputime-start_time;