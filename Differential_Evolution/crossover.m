function U=crossover(X,V,CR,crossStrategy)
[NP,Dim]=size(X);
switch crossStrategy
    %crossStrategy=1:binomial crossover
    case 1
        for i=1:NP
            jRand=randi([1,Dim]);%jRand∈[1,Dim]
            for j=1:Dim
                k=rand;
                if k<=CR||j==jRand %j==jRand是为了确保至少有一个U(i,j)=V(i,j)
                    U(i,j)=V(i,j);
                else
                    U(i,j)=X(i,j);
                end     
            end    
        end
    %crossStrategy=2:Exponential crossover
    case 2
        for i=1:NP
            j=randi([1,Dim]);%j∈[1,Dim]
            L=0;
            U(i,:)=X(i,:);
            k=rand;
            while(k<CR && L<Dim)
                U(i,j)=V(i,j);
                j=j+1;
                if(j>Dim)
                    j=1;
                end
                L=L+1;
            end
        end
    otherwise
        error('没有所指定的交叉策略，请重新设定crossStrategy的值');
end
        
