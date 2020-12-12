function [Outputs,Pre_Labels]=LIFT(train_data,train_target,test_data,test_target,ratio,svm)

    if(nargin<6)
        svm.type='Linear';
        svm.para=[];
    end
    
    if(nargin<5)
        ratio=0.1;
    end
    
    if(nargin<4)
        error('Not enough input parameters, please type "help LIFT" for more information');
    end
    
    [num_train,dim]=size(train_data);
    [num_class,num_test]=size(test_target);
    
    P_Centers=cell(num_class,1);
    N_Centers=cell(num_class,1);
    
    %Find key instances of each label
    for i=1:num_class
        disp(['Performing clusteirng for the ',num2str(i),'/',num2str(num_class),'-th class']);
        
        p_idx=find(train_target(i,:)==1);
        n_idx=setdiff([1:num_train],p_idx);
        
        p_data=train_data(p_idx,:);
        n_data=train_data(n_idx,:);
        
        k1=min(ceil(length(p_idx)*ratio),ceil(length(n_idx)*ratio));
        k2=k1;
        
        if(k1==0)
            POS_C=[];
            [NEG_IDX,NEG_C]=kmeans(train_data,min(50,num_train),'EmptyAction','singleton','OnlinePhase','off','Display','off');
        else
            if(size(p_data,1)==1)
                POS_C=p_data;
            else
                [POS_IDX,POS_C]=kmeans(p_data,k1,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end
            
            if(size(n_data,1)==1)
                NEG_C=n_data;
            else
                [NEG_IDX,NEG_C]=kmeans(n_data,k2,'EmptyAction','singleton','OnlinePhase','off','Display','off');
            end
        end  
        
        P_Centers{i,1}=POS_C;
        N_Centers{i,1}=NEG_C;
    end
    
    switch svm.type
        case 'RBF'
            gamma=num2str(svm.para);
            str=['-t 2 -g ',gamma,' -b 1'];
        case 'Poly'
            gamma=num2str(svm.para(1));
            coef=num2str(svm.para(2));
            degree=num2str(svm.para(3));
            str=['-t 1 ','-g ',gamma,' -r ', coef,' -d ',degree,' -b 1'];
        case 'Linear'
            str='-t 0 -b 1';
        otherwise
            error('SVM types not supported, please type "help LIFT" for more information');
    end
    
    Models=cell(num_class,1);
    
    %Perform representation transformation and training
    for i=1:num_class        
        disp(['Building classifiers: ',num2str(i),'/',num2str(num_class)]);
        
        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);
        
        data=[];
        
        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        else
            blocksize=5000-num_center;
            num_block=ceil(num_train/blocksize);
            for j=1:num_block-1
                low=(j-1)*blocksize+1;
                high=j*blocksize;
                
                tmp_mat=[centers;train_data(low:high,:)];
                Y=pdist(tmp_mat);
                Z=squareform(Y);
                data=[data;Z((num_center+1):(num_center+blocksize),1:num_center)];                
            end
            
            low=(num_block-1)*blocksize+1;
            high=num_train;
            
            tmp_mat=[centers;train_data(low:high,:)];
            Y=pdist(tmp_mat);
            Z=squareform(Y);
            data=[data;Z((num_center+1):(num_center+high-low+1),1:num_center)];
        end
        
        training_instance_matrix=data;
        training_label_vector=train_target(i,:)';
        
        Models{i,1}=svmtrain(training_label_vector,training_instance_matrix,str);      
    end
    
    %Perform representation transformation and testing
    Pre_Labels=[];
    Outputs=[];
    for i=1:num_class        
        centers=[P_Centers{i,1};N_Centers{i,1}];
        num_center=size(centers,1);
        
        data=[];
        
        if(num_center>=5000)
            error('Too many cluster centers, please try to decrease the number of clusters (i.e. decreasing the value of ratio) and try again...');
        else
            blocksize=5000-num_center;
            num_block=ceil(num_test/blocksize);
            for j=1:num_block-1
                low=(j-1)*blocksize+1;
                high=j*blocksize;
                
                tmp_mat=[centers;test_data(low:high,:)];
                Y=pdist(tmp_mat);
                Z=squareform(Y);
                data=[data;Z((num_center+1):(num_center+blocksize),1:num_center)];                
            end
            
            low=(num_block-1)*blocksize+1;
            high=num_test;
            
            tmp_mat=[centers;test_data(low:high,:)];
            Y=pdist(tmp_mat);
            Z=squareform(Y);
            data=[data;Z((num_center+1):(num_center+high-low+1),1:num_center)];
        end
        
        testing_instance_matrix=data;
        testing_label_vector=test_target(i,:)';
        
        [predicted_label,accuracy,prob_estimates]=svmpredict(testing_label_vector,testing_instance_matrix,Models{i,1},'-b 1');
        if(isempty(predicted_label))
            predicted_label=train_target(i,1)*ones(num_test,1);
            if(train_target(i,1)==1)
                Prob_pos=ones(num_test,1);
            else
                Prob_pos=zeros(num_test,1);
            end
            Outputs=[Outputs;Prob_pos'];
            Pre_Labels=[Pre_Labels;predicted_label'];
        else
            pos_index=find(Models{i,1}.Label==1);
            Prob_pos=prob_estimates(:,pos_index);
            Outputs=[Outputs;Prob_pos'];
            Pre_Labels=[Pre_Labels;predicted_label'];
        end
    end
    
