function [weights] = weight_Park2008_Binary(X, Y)
% X: N*d matrix for training vectors, where each row indicates a training isntance
% Y: N*q matrix for label vectors with +1/0;
% weights: N*q weight matrix.

% This weight form was used in 
% [1] Lee M, Park CH. On applying dimension reduction for multi-labeled problems. MLDM2007, LNAI4571, pp.131-143, 2007.
% [2] Park CH, Lee M.On applying dimension reduction for multi-labeled problems. PRL, 29:828-887, 2008.
% [3] Wang H, Ding C, Huang H. Multi-label linear discriminant analysis. ECCV2010, LNCS6316, pp.126-139, 2010.
% 

    weights = Y;
    
end