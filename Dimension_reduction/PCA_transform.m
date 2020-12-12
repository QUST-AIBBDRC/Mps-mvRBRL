function [P] = PCA_transform(X, Y, parameter)
% This function is to execute PCA
% Editor: Jianhua Xu (xujianhua@njnu.edu.cn)
% Date: May, 2015.

% Input: 
%       X -- centered training instance vectors (N *d)
%       parameter.ratio -- reduced ratio
% Output: 
%       P -- orthonomal projection matrix (d*d) 
%       X'*X*P= lambda*P
%----------------------------------------------------------
    [N,q]=size(Y);
    
    G= X'*X;
    
    if(norm(G-G','fro') ~= 0)
        disp('Warning: not a real symmetrical matrix !');
    end

    if(parameter.rank == 1)
        rankG = rank(G);
        disp(strcat('The rank of matrix =  ', num2str(rankG)));
    else
        rankG = size(X,1);
    end
    
    [V,D] = eig(G);
    
    eigenVectors=V;
    eigenValues=diag(D);
    
    [eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
    reduced_dimension = detect_reduced_dimension(eigenValues, rankG, q, parameter.ratio);
    
    P = eigenVectors(:, 1:reduced_dimension);

end
