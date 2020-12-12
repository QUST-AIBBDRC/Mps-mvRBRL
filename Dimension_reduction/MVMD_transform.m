function [P] = MVMD_transform(X, Y, parameter)

% Input: 
%       X -- centered training instance matrix (N *d)
%       Y -- centered training label matrix (N*q)
%       parameter.ratio -- reduced ratio
%       parameter.beta -- trade-off factor
% Output: 
%       P -- projection matrix (d*dim) 
%  [(1-beta)*X'*X+beta*(X'*H*Y*Y'*H*X)]*P = lambda*P

disp('MVMD (Xu et al, KBS, 2016) ..........................................');
[N1, d] = size(X);
[N2, q] = size(Y);
if(N1 ~= N2) disp('The number of training instances in X and Y is not equal'); end
N = min(N1, N2);

A=X'*X;

beta = parameter.beta;

H = eye(N,N) - 1/N;

%L = Y*Y';
%B = X'*(H*L*H)*X;
%B = (B+B')/2.0;

XHY = X'*H*Y;
B = XHY*XHY';

%beta0= balanced_factor_MVMD(A,B,3);

norm_fro = norm(A-A', 'fro');
if (norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix A= ', num2str(norm_fro)));
end

norm_fro = norm(B-B', 'fro');
if (norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix B = ', num2str(norm_fro)));
end

%beta00 = beta0
beta00 = beta;

if(beta00<0.0)beta00=0.0;end
if(beta00>1.0)beta00=1.0;end
disp(strcat('Beta value = ', num2str(beta00)));

G = (1-beta00)*A + beta00*B;

norm_fro = norm(G-G', 'fro');
if (norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix = ', num2str(norm_fro)));
end

if(parameter.rank==1)
    rankG = rank(G);
    disp(strcat('The rank of matrix G = ',num2str(rankG)));
else
    rankG = N;
end

[V, D] = eig(G);

eigenVectors = V;
eigenValues= diag(D);

[eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
reduced_dimension = detect_reduced_dimension(eigenValues, rankG, q, parameter.ratio);
    
P = eigenVectors(:, 1:reduced_dimension);

end
