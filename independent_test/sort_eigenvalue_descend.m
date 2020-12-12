function [eigenValues, eigenVectors] = sort_eigenvalue_descend(eigenValues, eigenVectors)
% this function sorts eigenvalues in descending order and their corresponding eigenvectors.
% Editor: Jianhua Xu (xujianhua@njnu.edu.cn)
% Date: May, 2015

% To check all eigenvalues, which are divided into
% --Real eigenvalues
%   -- positive
%   -- negative
%   -- infitinte (+infinite and -infinite
% --Complex eigenvalues
% Input:
%      eigenValuses  -- eigenvalues  (D*1)
%      eigenVectors  -- eigenvectors (D*D), where each column corresponds
%                       to a projection direction
% Output: processed eigenvalues and eigenvectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    nr_dimension = size(eigenValues);

    nr_rpositive =0;
    nr_rnegative =0;
    nr_rinfinite=0;
    nr_real=0;
    nr_complex=0;
    nr_nan=0;
    for (i=1:nr_dimension)
        if(isreal(eigenValues(i))==1) %real eigenvalue
            nr_real = nr_real + 1;
            if(eigenValues(i) > 0) nr_rpositive = nr_rpositive + 1; end
            if(eigenValues(i) <= 0) nr_rnegative = nr_rnegative + 1; end
            if(isinf(eigenValues(i))) nr_rinfinite = nr_rinfinite + 1; end    
            if(isnan(eigenValues(i))) nr_nan = nr_nan +1; end     
        else % complex vigenvalue
            nr_complex = nr_complex + 1;
        end
    end
    
    disp('Status of eigenvalues');
    disp(strcat('The number of real eigenvalues =  ',num2str(nr_real)));
    disp(strcat('----positive  =  ',num2str(nr_rpositive)));
    disp(strcat('----negative  =  ',num2str(nr_rnegative)));
    disp(strcat('----infinite  =  ',num2str(nr_rinfinite)));
    disp(strcat('----Nan       =  ',num2str(nr_nan)));
    disp(strcat('The number of complex eigenvalues =  ',num2str(nr_complex)));

    for (i=1:nr_dimension)
        if((isreal(eigenValues(i))==1) & (eigenValues(i) > 0.0) & (isinf(eigenValues(i)) == 0)) %real positive eigenvalue
            continue;
        end
        eigenValues(i)=0.0;
    end

    
    [eigenValues order] = sort(eigenValues, 'descend');
        
    eigenVectors = eigenVectors(:,order);

 
end