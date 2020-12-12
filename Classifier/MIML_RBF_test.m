function [Outputs,Pre_Labels]=MIML_RBF_test(test_bags,test_target,Centroids,Sigma_value,Weights)
%MIML_RBF_test tests a multi-instance multi-label RBF learner.
%
%    Syntax
%
%       [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision,Outputs,Pre_Labels,te_time]=MIML_RBF_test(test_bags,test_target,Centroids,Sigma_value,Weights,ratio,mu)
%
%    Description
%
%       LML_test takes,
%           test_bags        - An M2x1 cell, the ith test bag is stored in test_bags{i,1}
%           test_target      - A QxM2 array, if the ith test bag belongs to the jth class, test_target(j,i) equals +1, otherwise test_target(j,i) equals -1
%           Centroids        - A Kx1 cell structure, where the k-th centroid of the RBF neural network is stored in Centroid{k,1}
%           Sigma_value      - A 1xK vector, where the sigma value for the k-th centroid is stored in Sigma_value(1,k)
%           Weights          - A QxQ matrix used for label prediction
%           ratio            - The number of centroids of the i-th class is set to be ratio*Ti, where Ti is the number of train bags with lable i
%           mu               - The ratio used to determine the standard deviation of the Gaussian activation function
%      and returns,
%           HammingLoss       - The hamming loss on testing data
%           RankingLoss       - The ranking loss on testing data
%           OneError          - The one-error on testing data as
%           Coverage          - The coverage on testing data as
%           Average_Precision - The average precision on testing data
%           Outputs           - A QxM2 array, the probability of the ith testing instance belonging to the jCth class is stored in Outputs(j,i)
%           Pre_Labels        - A QxM2 array, if the ith testing instance belongs to the jth class, then Pre_Labels(j,i) is +1, otherwise Pre_Labels(j,i) is -1
%           te_time           - The time spent in testing

    start_time=cputime;
    
    [num_class,num_test]=size(test_target);
    num_centroid=size(Centroids,1);
    
    A=zeros(num_test,num_centroid+1);
    
    for i=1:num_test
        if(mod(i,10)==0)
            disp(strcat(num2str(i),'/',num2str(num_test)));
        end
        
        temp_vec=zeros(1,num_centroid);
        for k=1:num_centroid
            tmp=GMIL_Hausdorff(test_bags{i,1},Centroids{k,1});
            temp_sigma=Sigma_value(1,k);
            temp_vec(1,k)=exp(-tmp^2/(2*temp_sigma^2));
        end
        temp_vec=[1,temp_vec];
        
        A(i,:)=temp_vec;
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
