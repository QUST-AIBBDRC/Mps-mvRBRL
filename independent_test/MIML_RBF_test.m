function [Outputs,Pre_Labels]=MIML_RBF_test(test_bags,test_target,Centroids,Sigma_value,Weights)

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
  