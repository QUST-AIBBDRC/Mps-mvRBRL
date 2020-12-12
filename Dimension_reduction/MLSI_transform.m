function [P] = MLSI_transform(X, Y, parameter)
% 
% Input: 
%       X -- centered training instance matrix (N *d)
%       Y -- centered training label matrix (N*q)
%       parameter.ratio -- reduced ratio
%       parameter.beta -- trade-off factor (0,1)
%       parameter.regX -- regularization constant
% Output: 
%       P -- orthonomal projection matrix (d*dim) 
%    
% Reference:
% Kai Yu, Shipeng Yu, Volker Tresp. Multi-label Informed Latent Semantic
%                                   Indexing. SIGIR2005, 258-265.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('MLSI.......................................');

[N1, d] = size(X);
[N2, q] = size(Y);
if(N1 ~= N2) disp('The number of training instances in X and Y is not equal'); end
N =min(N1, N2);

beta = parameter.beta;
regxy = parameter.regXY;
regx = parameter.regX;

C = (1 - beta) * X * X' + beta * Y * Y';
maxC = max(max(C));
minC = min(min(C));
disp(strcat('C: ', num2str(maxC), ' to ', num2str(minC)));
if (regxy <0)
    C = C - regxy*maxC*eye(N,N);
else % regxy >=0
    C = C + regxy*eye(N,N);
end

if (regxy == 0)
    S = X' * pinv(C) * X;
else
    S = X'*inv(C)*X;
end

maxS = max(max(S));
minS = min(min(S));
disp(strcat('S: ', num2str(maxS), ' to ', num2str(minS)));
if (regx < 0)
    S = S - regx*maxS*eye(d,d);
else %regx>=0)
    S = S + regx*eye(d,d);
end

S= (S+S')/2.0;

M = X' * X;
M = (M+M')/2;

norm_fro = norm(S-S','fro');
if(norm_fro ~= 0)
    disp('Warning: not a symmetrical matrix !');
end

if(parameter.rank == 1)
    rankM = rank(M);
    rankS = rank(S);
    disp(strcat('The ranks of two matrixes = ',num2str(rankM),', ',  num2str(rankS)));
    min_rank = min(rankM, rankS);
else
    min_rank = N;
end

[V, D] = eig(M,S);

%V'*V

eigenVectors = V;
eigenValues= diag(D);

[eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
reduced_dimension = detect_reduced_dimension(eigenValues, min_rank, q, parameter.ratio);
    
P = eigenVectors(:, 1:reduced_dimension);

end