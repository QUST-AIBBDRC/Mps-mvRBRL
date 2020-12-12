function [P] = wMLDA_transform(X, Y, parameter,weight_type)

disp('Concise MLDA ..................');

[W] = weight_estimation(X, Y, weight_type);

[Nx,d] = size(X); % d--the number of features
[Ny,q] = size(Y); % q--the number of labels

if(Nx ~= Ny) disp('Warning: check the training data');end
[N] = min([Nx,Ny]);

% the sum of weight for each label and the total sum of weights
% Nk -- a q-dimensonal row vector
Nk = sum(W);
Nt = sum(Nk);

%N1=1./Nk;
for i=1:q
    if(Nk(i)==0)
        N1(i)=0;
    else
        N1(i) = 1/Nk(i);
    end
end

sumW = sum(W')';
Hw = W*diag(N1)*W';
Hb = Hw;
Hw = -Hw + diag(sumW);
Hb = Hb - sumW*sumW'/Nt;

Sw=X'*Hw*X;
Sb=X'*Hb*X;

Sw = (Sw + Sw')/2.0;
Sb = (Sb + Sb')/2.0;
%maxSw = max(max(Sw));
%minSw = min(min(Sw));
maxSw = max(diag(Sw));
minSw = min(diag(Sw));
disp(strcat('Sw maximal component = ',num2str(maxSw),', minimal component = ',num2str(minSw)));

regXY = parameter.regXY
if (regXY < 0)
    Sw = Sw - regXY*maxSw*eye(d,d);
else % >=0
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

if(parameter.rank == 1)
    rankSb = rank(Sb);
    rankSw = rank(Sw);
    disp(strcat('The ranks of two matrixes (Sb and Sw) = ',num2str(rankSb), ', ',  num2str(rankSw)));
    min_rank = min(rankSb, rankSw);
else
    min_rank = q-1;
end


[V D] = eig(Sb,Sw);

eigenVectors = V;
eigenValues= diag(D);

[eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors);
    
reduced_dimension = detect_reduced_dimension(eigenValues, min_rank, q, parameter.ratio);
    
P = eigenVectors(:, 1:reduced_dimension);

end