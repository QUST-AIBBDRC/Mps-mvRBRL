function [P] = execute_transform(X, Y, transform_type, transform_parameter)
%
parameter = transform_parameter;

switch(transform_type)
    case 1 %PCA
        disp('PCA -------------------');
        [P] = PCA_transform(X,Y,parameter);

 
    case 2 %MLSI
        disp('MLSI--------------------');
        [P] = MLSI_transform(X,Y,parameter);
        

        
    case 3 %MDDMf with orthonormal features
        disp('MDDMf--------------------');
        [P] = MDDM_transform(X,Y,parameter,1);
        
    case 4 %MVMD with orthonormal transforms
        disp('MVMD--------------------');
        [P] = MVMD_transform(X,Y,parameter);   

      
    case 5 %wMLDAb with binary weight form
        disp('wMLDAb--------------------');
        [P] = wMLDA_transform(X,Y,parameter,1);
        
   
end

end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
