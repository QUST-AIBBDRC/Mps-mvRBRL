function [dim] = detect_reduced_dimension(eigenValues, min_rank, num_class, ratio)
%This function is determine the effective eigen-vectors
%Editor: Jianhua Xu
%Data: May, 2015
%
% Input: 
%      eigenValues -- eigenvalues
%      min_rank -- effective rank
%      num_class -- the number of classes
%      ratio -- ratio to be reduced [0,length(eigenValues)]
%            =-1 -- dim = #classes
%            =-1 -- dim = #classes-1
%            =0 -- dim = non_zeros, that is, the number of all 
%                       non-zero eigenvalues (>=1.0e-10*max(eigenValues)
%            =(0,1.0) -- dim = ratio*length(eigenValues)
%            =(1, 2.0) -- dim = argmin sum_i^dim eigenValues(i)/sum(eigenValues)
%                       >= ratio-1.0     
%            =1 or >=2.0 -- dim =ratio
%      Note: dim <= the number of non-zero eigenvalues
% Output: dim -- reduced dimension
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(nargin<3) ratio = 0.1; end
%-------------------------------------------------------------------------
    %eigenValues
 
    eigenValues = eigenValues/max(eigenValues);
    non_zeros = sum(eigenValues >= 1.0e-10);
    disp(strcat('The number of real positive eigenvalues (1.0e-10)= ', num2str(non_zeros)));
    if(min_rank < non_zeros) non_zeros = min_rank; end
    disp(strcat('The number of true real positive eigenvalues = ', num2str(non_zeros)));

    nr_dimension = length(eigenValues);
    
    disp(strcat('The number of original data dimensions      = ', num2str(nr_dimension)));

    if(ratio == -2) dim= num_class; end
    
    if(ratio == -1) dim = num_class -1 ; end
    
    if(ratio ==1 | ratio >=2 )
        dim = ratio;
    end
    
    if(ratio == 0)
        dim= non_zeros;
    end
        
    if (ratio >0 & ratio <1)
            dim = ceil(ratio*nr_dimension);
    end
            
    if (ratio > 1 & ratio < 2)
            sum_eigen=0.0;
            sum_all = sum(eigenValues(1:non_zeros));
            percentage = ratio - 1.0;
            for i=1:non_zeros
                sum_eigen = sum_eigen + eigenValues(i);
                dim = i;
                if(sum_eigen/sum_all >= percentage) break; end
            end
            
    end

    disp(strcat('The number of reduced dimensions      = ', num2str(dim)));
end  
% =========================================================================