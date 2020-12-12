function [P] = DMLDA_transform(X, Y, parameter, weight_type)
%Oikonomou M, Tefas A. Direct multi-label linear discriminant analysis. EANN 2013,CCIS3883, 414-423, 2013
%This function is to execute direct multi-label LDA.
%Editor: Jianhua Xu(xujianhua@njnu.edu.cn)
%Date: April, 2015.
% Input: 
%       X -- centered training instance matrix (N *d)
%       Y -- training label matrix (N*q) (+1/-1)
%       parameter.ratio -- reduced ratio
%       parameter.regXY -- constant for X and Y
% Output: 
%       P -- projection matrix (d*dim) 
%    Sb*P=lambda*(Sw+RegXY*I)*P
%--------------------------------------------------------------------------
disp('Concise DMLDA ........................................');
%determine the weighted matrix
[W] = weight_estimation(X, Y, weight_type);
[Nx,d] = size(X);
[Ny,q] = size(Y);

if(Nx ~= Ny) disp('Warning: the number of training instances in X & Y is not equal');end
[N] = min([Nx,Ny]);
    
% the sum of weight for each label and the total sum of weights
% Nk -- a q-dimensonal row vector
Nk = sum(W);
Nt = sum(Nk);
%N1 = 1./Nk;
for i=1:q
    if(Nk(i)==0)
        N1(i)=0;
    else
        N1(i) = 1/Nk(i);
    end
end

N2 = N1.*N1;
sumW = sum(W')';

Hw = diag(sumW)-W*diag(N1)*W';

Hb1 = q*eye(N,N)-2*ones(N,1)*N1*W'+N*W*diag(N2)*W';

Sw=X'*Hw*X;
Sb=X'*Hb1*X-Sw;

Sw = (Sw + Sw')/2.0;
Sb = (Sb + Sb')/2.0;

regXY = parameter.regXY;
if(regXY <0)
    maxSw = max(max(Sw));
    Sw = Sw - regXY*maxSw*eye(d,d);
else %regXY >= 0
    Sw = Sw + regXY*eye(d,d);
end

norm_fro = norm(Sw-Sw', 'fro');
if(norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix (Sw) = ',num2str(norm_fro)));
end

norm_fro = norm(Sb-Sb', 'fro');
if(norm_fro ~= 0)
    disp(strcat('Warning: not a real symmetrical matrix (Sb) = ',num2str(norm_fro)));
end

%Sb
%Sw
if(parameter.rank==1)
    rankSb = rank(Sb);
    rankSw = rank(Sw);
    disp(strcat('The ranks of two matrixes = ',num2str(rankSb),', ',  num2str(rankSw)));
    min_rank = min(rankSb, rankSw);
else
    min_rank=d;
end

[V, D] = eig(Sb,Sw); 

eigenVectors = V;
eigenValues= diag(D);

[eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
reduced_dimension = detect_reduced_dimension(eigenValues, min_rank, q, parameter.ratio);
    
P = eigenVectors(:, 1:reduced_dimension);

end