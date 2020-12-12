function ys=gaussian_kern_reg(xs,x,y,h)
% xs -- An N*1 array,
% x -- An N*M array,  the ith instance of training instance is stored in x(:,i)
% y -- An A QxM array,
% Gaussian kernel function
K=sqdist(diag(1./h)*x,diag(1./h)*xs);
K=exp(-K/2);

    ys=(y*K)./sum(K,1);
    
% end

% Gaussian kernel function
% K1=sqdist(diag(1./h)*x,diag(1./h)*xs);
% K1=exp(-K1/2);
% 
% % linear kernel regression
% ys1=sum(K1'.*y)/sum(K1);
% YS2=ys1;