function [recall, precision, rate] = recall_precision(Wtrue, Dhat)
%

max_hamm = max(Dhat(:))
hamm_thresh = min(3,max_hamm);

[Ntest, Ntrain] = size(Wtrue);
total_good_pairs = sum(Wtrue(:));

% find pairs with similar codes
precision = zeros(max_hamm,1);
recall = zeros(max_hamm,1);
rate = zeros(max_hamm,1);

for n = 1:length(precision)
    j = (Dhat<=((n-1)+0.00001));

    %exp. # of good pairs that have exactly the same code
    retrieved_good_pairs = sum(Wtrue(j));

    % exp. # of total pairs that have exactly the same code
    retrieved_pairs = sum(j(:));

    precision(n) = retrieved_good_pairs/retrieved_pairs;
    recall(n)= retrieved_good_pairs/total_good_pairs;
    rate(n) = retrieved_pairs / (Ntest*Ntrain);
end
