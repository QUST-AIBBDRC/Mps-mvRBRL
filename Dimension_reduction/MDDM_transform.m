function [P] = MDDM_transform(X,Y, parameter, project_type)
% This function is from Zhi-Hua Zhou homepage and is modified by Jianhua Xu
% Input: 
%       X -- centered training instance matrix (N *d)
%       Y -- centered training label matrix (N*q)
%       parameter.ratio -- reduced ratio
%       project_type: 0-->MDDM/1-->MDDMf
% Output: 
%       P -- projection matrix (d*dim) 
% MDDMp: [X'*H*(Y*Y')*H*X]*P=lambda*P
% MDDMf: [X'*H*(Y*Y')*H*X]*P=lambda*[beta*X'*X+(1-beta)*I]*P

[N1, d] = size(X);
[N2, q] = size(Y);
if(N1 ~= N2) disp('The number of training instances in X and Y is not equal'); end
N =min(N1, N2);

H=eye(N,N)-1/N;

XHY = X'*H*Y;
G= XHY*XHY';

norm_fro = norm(G-G', 'fro');
if(norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix = ', num2str(norm_fro)));
end 

if(parameter.rank == 1)
    rankG = rank(G);
    disp(strcat('The rank of matrix = ',num2str(rankG)));
else
    rankG = q;
end

if(project_type == 0)
    [V, D] = eig(G);
else
    mu = parameter.beta;
    B = mu*X'*X + (1-mu)*eye(d,d);
    [V, D] = eig(G,B);
end

eigenVectors = V;
eigenValues= diag(D);

[eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
reduced_dimension = detect_reduced_dimension(eigenValues, rankG, q, parameter.ratio);
    
P = eigenVectors(:, 1:reduced_dimension);

end
