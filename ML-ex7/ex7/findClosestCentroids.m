function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


% cluster assignment step

m = size(X, 1);

% matrix store distance from each example to cluster centroids
c = zeros(size(X,1), 3);

for i = 1 : m,
    for j = 1 : K,
        c(i,j) = ((X(i,:) - centroids(j,:)) * (X(i,:) - centroids(j,:))');
    end;
end;

[~,idx] = min(c,[],2);



% for i = 1:length(idx)
%     % repmat replicates the same training sample into matrix M x N: repmat(A, M, N)
%     % this will help us compute distances and errors vectorised way
%     distance = (repmat(X(i,:), K, 1) - centroids).^2;
%     errors = sum(distance, 2).^2;
%     [~, idx(i)] = min(errors);
% end



% =============================================================

end

